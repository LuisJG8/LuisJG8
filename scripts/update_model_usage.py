#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

README_START = "<!-- MODEL_USAGE:START -->"
README_END = "<!-- MODEL_USAGE:END -->"
DEFAULT_COMMAND = [
    "npx",
    "--yes",
    "@ccusage/codex@latest",
    "daily",
    "--json",
    "--breakdown",
    "--offline",
]


@dataclass
class ModelSummary:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    reasoning_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate GitHub profile model usage from @ccusage/codex."
    )
    parser.add_argument("--readme", default="README.md", help="README file path.")
    parser.add_argument(
        "--json-output",
        default="data/model-usage.json",
        help="Output path for machine-readable summary JSON.",
    )
    parser.add_argument(
        "--input-json",
        help="Optional local JSON file (for testing) instead of running npx.",
    )
    parser.add_argument("--since", help="Optional date for ccusage (YYYYMMDD or YYYY-MM-DD).")
    parser.add_argument("--until", help="Optional date for ccusage (YYYYMMDD or YYYY-MM-DD).")
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds for the npx command.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    command = DEFAULT_COMMAND.copy()
    if args.since:
        command.extend(["--since", args.since])
    if args.until:
        command.extend(["--until", args.until])
    return command


def read_source_json(args: argparse.Namespace) -> dict[str, Any]:
    if args.input_json:
        with Path(args.input_json).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise SystemExit("Input JSON must be an object at the top level.")
        return payload

    command = build_command(args)
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            "Unable to run `npx`. Install Node.js and npm on the runner."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SystemExit(
            "Timed out waiting for @ccusage/codex.\n"
            f"Command: {' '.join(command)}\n"
            f"Timeout: {args.timeout}s"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip() or "No stderr output captured."
        raise SystemExit(
            "Failed to collect Codex usage data.\n"
            f"Command: {' '.join(command)}\n"
            f"Exit code: {result.returncode}\n"
            f"stderr:\n{stderr}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        preview = result.stdout[:500]
        raise SystemExit(
            "The @ccusage/codex command did not return valid JSON.\n"
            f"stdout preview:\n{preview}"
        ) from exc

    if not isinstance(payload, dict):
        raise SystemExit("Unexpected response: top-level JSON is not an object.")
    return payload


def rows_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def normalize_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = rows_from_value(payload.get("data"))
    if rows:
        return rows

    rows = rows_from_value(payload.get("daily"))
    if rows:
        return rows

    projects = payload.get("projects")
    if isinstance(projects, dict):
        merged: list[dict[str, Any]] = []
        for project_payload in projects.values():
            if isinstance(project_payload, dict):
                merged.extend(rows_from_value(project_payload.get("data")))
                merged.extend(rows_from_value(project_payload.get("daily")))
            else:
                merged.extend(rows_from_value(project_payload))
        if merged:
            return merged

    return []


def get_number(payload: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def iter_model_breakdowns(row: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    for key in ("breakdown", "modelBreakdowns"):
        breakdown = row.get(key)
        if isinstance(breakdown, dict):
            return [
                (str(model), stats)
                for model, stats in breakdown.items()
                if isinstance(stats, dict)
            ]
        if isinstance(breakdown, list):
            parsed: list[tuple[str, dict[str, Any]]] = []
            for item in breakdown:
                if not isinstance(item, dict):
                    continue
                model = item.get("model") or item.get("name")
                if isinstance(model, str):
                    parsed.append((model, item))
            if parsed:
                return parsed

    models_dict = row.get("models")
    if isinstance(models_dict, dict):
        parsed = [
            (str(model), stats)
            for model, stats in models_dict.items()
            if isinstance(stats, dict)
        ]
        if parsed:
            return parsed

    model_value = row.get("model")
    if isinstance(model_value, str):
        return [(model_value, row)]

    models = row.get("models") or row.get("modelsUsed") or []
    if isinstance(models, list) and len(models) == 1 and isinstance(models[0], str):
        return [(models[0], row)]
    return []


def serialize_model(item: ModelSummary) -> dict[str, Any]:
    payload = asdict(item)
    payload["cost_usd"] = round(item.cost_usd, 6)
    return payload


def normalize_date(date_text: str) -> tuple[datetime | None, str]:
    date_text = date_text.strip()
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            parsed = datetime.strptime(date_text, fmt)
            return parsed, parsed.date().isoformat()
        except ValueError:
            continue
    return None, date_text


def build_summary(rows: list[dict[str, Any]], source_command: str) -> dict[str, Any]:
    per_model: dict[str, ModelSummary] = {}
    parsed_period_dates: list[datetime] = []
    raw_period_dates: list[str] = []

    for row in rows:
        date_value = row.get("date")
        if isinstance(date_value, str):
            parsed, normalized = normalize_date(date_value)
            raw_period_dates.append(normalized)
            if parsed is not None:
                parsed_period_dates.append(parsed)

        model_rows = iter_model_breakdowns(row)
        row_cost = get_number(row, "costUSD", "totalCost", "cost_usd")
        prepared_models: list[dict[str, Any]] = []

        for model_name, stats in model_rows:
            model = per_model.setdefault(model_name, ModelSummary(model=model_name))
            input_tokens = int(get_number(stats, "inputTokens", "input_tokens"))
            output_tokens = int(get_number(stats, "outputTokens", "output_tokens"))
            cache_creation_tokens = int(
                get_number(stats, "cacheCreationTokens", "cache_creation_tokens")
            )
            cache_read_tokens = int(
                get_number(stats, "cacheReadTokens", "cache_read_tokens")
            )
            cached_input_tokens = int(get_number(stats, "cachedInputTokens"))

            # In Codex daily models dict format, inputTokens includes cachedInputTokens.
            if cache_read_tokens == 0 and cached_input_tokens > 0:
                cache_read_tokens = cached_input_tokens
                if input_tokens >= cached_input_tokens:
                    input_tokens -= cached_input_tokens

            reasoning_tokens = int(
                get_number(
                    stats,
                    "reasoningTokens",
                    "reasoningOutputTokens",
                    "reasoning_tokens",
                )
            )

            total = int(get_number(stats, "totalTokens", "total_tokens"))
            if total == 0:
                total = (
                    input_tokens
                    + output_tokens
                    + cache_creation_tokens
                    + cache_read_tokens
                )
            stats_cost = get_number(stats, "costUSD", "totalCost", "cost_usd")
            prepared_models.append(
                {
                    "summary": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "cache_creation_tokens": cache_creation_tokens,
                    "cache_read_tokens": cache_read_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "total_tokens": total,
                    "stats_cost": stats_cost,
                }
            )

        if not prepared_models:
            continue

        sum_stats_cost = sum(item["stats_cost"] for item in prepared_models)
        allocated_costs: list[float] = []
        if row_cost > 0 and sum_stats_cost == 0:
            if len(prepared_models) == 1:
                allocated_costs = [row_cost]
            else:
                weight_total = sum(item["total_tokens"] for item in prepared_models)
                if weight_total > 0:
                    allocated_costs = [
                        row_cost * (item["total_tokens"] / weight_total)
                        for item in prepared_models
                    ]
                else:
                    equal_cost = row_cost / len(prepared_models)
                    allocated_costs = [equal_cost for _ in prepared_models]
        else:
            allocated_costs = [item["stats_cost"] for item in prepared_models]

        for item, allocated_cost in zip(prepared_models, allocated_costs, strict=True):
            model = item["summary"]
            model.input_tokens += item["input_tokens"]
            model.output_tokens += item["output_tokens"]
            model.cache_creation_tokens += item["cache_creation_tokens"]
            model.cache_read_tokens += item["cache_read_tokens"]
            model.reasoning_tokens += item["reasoning_tokens"]
            model.total_tokens += item["total_tokens"]
            model.cost_usd += allocated_cost

    models = sorted(
        per_model.values(),
        key=lambda item: (-item.total_tokens, item.model.lower()),
    )

    totals = ModelSummary(model="TOTAL")
    for item in models:
        totals.input_tokens += item.input_tokens
        totals.output_tokens += item.output_tokens
        totals.cache_creation_tokens += item.cache_creation_tokens
        totals.cache_read_tokens += item.cache_read_tokens
        totals.reasoning_tokens += item.reasoning_tokens
        totals.total_tokens += item.total_tokens
        totals.cost_usd += item.cost_usd

    if parsed_period_dates:
        period_from = min(parsed_period_dates).date().isoformat()
        period_to = max(parsed_period_dates).date().isoformat()
    else:
        period_from = min(raw_period_dates) if raw_period_dates else None
        period_to = max(raw_period_dates) if raw_period_dates else None

    return {
        "generatedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "period": {"from": period_from, "to": period_to},
        "totals": serialize_model(totals),
        "models": [serialize_model(item) for item in models],
        "sourceCommand": source_command,
        "rowCount": len(rows),
    }


def format_int(value: int) -> str:
    return f"{value:,}"


def format_usd(value: float) -> str:
    return f"${value:,.2f}"


def render_usage_block(summary: dict[str, Any]) -> str:
    generated_at = str(summary["generatedAt"]).replace("+00:00", "Z")
    period = summary["period"]
    models = summary["models"]
    totals = summary["totals"]

    lines = [
        '## <img src="codex-color.svg" alt="Codex logo" width="20" /> Codex Model Spend',
        "",
        f"_Last updated: {generated_at}_",
    ]
    if period.get("from") and period.get("to"):
        lines.append(f"_Coverage: {period['from']} to {period['to']}_")
    lines.append("")

    if not models:
        lines.append("No usage data found yet from local Codex logs.")
        return "\n".join(lines)

    lines.extend(
        [
            f"Tracked **{format_int(int(totals['total_tokens']))}** tokens across **{len(models)}** model(s), estimated spend **{format_usd(float(totals['cost_usd']))}**.",
            "",
            "| Model | Input tokens | Output tokens | Total tokens | Estimated cost |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for item in models:
        lines.append(
            f"| `{item['model']}` | {format_int(int(item['input_tokens']))} | {format_int(int(item['output_tokens']))} | {format_int(int(item['total_tokens']))} | {format_usd(float(item['cost_usd']))} |"
        )
    return "\n".join(lines)


def update_readme(readme_path: Path, usage_block: str) -> None:
    if readme_path.exists():
        content = readme_path.read_text(encoding="utf-8")
    else:
        content = "# Profile\n\n"

    replacement = f"{README_START}\n{usage_block}\n{README_END}"
    if README_START in content and README_END in content:
        before, remainder = content.split(README_START, maxsplit=1)
        _, after = remainder.split(README_END, maxsplit=1)
        updated = f"{before}{replacement}{after}"
    else:
        updated = content.rstrip() + f"\n\n{replacement}\n"
    readme_path.write_text(updated.rstrip() + "\n", encoding="utf-8")


def write_summary_json(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    payload = read_source_json(args)
    rows = normalize_rows(payload)
    summary = build_summary(rows, " ".join(build_command(args)))

    readme_path = Path(args.readme)
    json_path = Path(args.json_output)
    update_readme(readme_path, render_usage_block(summary))
    write_summary_json(json_path, summary)

    print(f"Updated {readme_path} and {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
