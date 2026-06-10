"""CLI entry point for ane-fp16-lint."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="ane-fp16-lint",
        description=(
            "Detect silent fp16 overflow/underflow in Core ML models "
            "before they corrupt inference on Apple Neural Engine."
        ),
    )
    parser.add_argument(
        "model",
        help="Path to a .mlpackage file to lint.",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply stable decompositions to unsafe ops (saves fixed model).",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save the fixed model (used with --fix).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output findings as JSON.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show stable alternative for each finding.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )

    args = parser.parse_args()

    try:
        from ane_fp16_lint import lint_model
    except ImportError as e:
        print(
            f"Error: Could not import ane_fp16_lint. "
            f"Make sure coremltools is installed: {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        report = lint_model(args.model, fix=args.fix)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        report.print(verbose=args.verbose)

    # Exit with non-zero if critical findings
    if report.critical_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
