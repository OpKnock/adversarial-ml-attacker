"""Command-line interface for the adversarial ML toolkit."""

from __future__ import annotations

import argparse
import sys

from .report import render_markdown, run_campaign, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="advml",
        description="Educational adversarial machine learning toolkit.",
    )
    subparsers = parser.add_subparsers(dest="command")

    campaign = subparsers.add_parser(
        "campaign", help="run the full robustness evaluation"
    )
    campaign.add_argument("--seed", type=int, default=42)
    campaign.add_argument("--out", default="reports")

    args = parser.parse_args(argv)
    if args.command == "campaign":
        results = run_campaign(seed=args.seed)
        write_report(results, args.out)
        print(render_markdown(results))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
