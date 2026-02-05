"""Command-line interface for MERGE-AI experiments."""

import argparse
import sys
from pathlib import Path

from .config import validate_api_keys, RESULTS_DIR, CONFLICTS_DIR
from .core.resolver import resolve_conflict
from .core.constraints import Constraint
from .evaluation.benchmark import Benchmark, run_benchmark
from .evaluation.experiments import load_conflicts_from_dir
from .evaluation.report import generate_report, generate_csv_summary
from .evaluation.error_analysis import analyze_all_hypotheses, save_error_analysis


def cmd_resolve(args):
    """Resolve a single merge conflict."""
    folder = Path(args.folder)

    # Load conflict files
    base_file = folder / "base.py"
    ours_file = folder / "ours.py"
    theirs_file = folder / "theirs.py"

    if not all(f.exists() for f in [base_file, ours_file, theirs_file]):
        print(f"Error: Missing required files in {folder}")
        print("Expected: base.py, ours.py, theirs.py")
        return 1

    base = base_file.read_text(encoding="utf-8")
    ours = ours_file.read_text(encoding="utf-8")
    theirs = theirs_file.read_text(encoding="utf-8")

    # Parse constraints
    constraints = []
    if args.constraints:
        constraint_map = {
            "ast": Constraint.AST,
            "imports": Constraint.IMPORTS,
            "functions": Constraint.FUNCTIONS,
            "all": Constraint.ALL,
        }
        for c in args.constraints.split(","):
            c = c.strip().lower()
            if c in constraint_map:
                constraints.append(constraint_map[c])

    print(f"Resolving conflict in {folder}...")
    if constraints:
        print(f"Constraints: {[c.value for c in constraints]}")

    result = resolve_conflict(
        base=base,
        ours=ours,
        theirs=theirs,
        constraints=constraints,
        max_retries=args.max_retries,
    )

    print(f"\nResolution {'successful' if result.success else 'failed'}")
    print(f"Attempts: {result.attempts}")
    print(f"Cost: ${result.cost_usd:.4f}")

    if result.errors_encountered:
        print(f"Errors encountered: {len(result.errors_encountered)}")
        for error in result.errors_encountered:
            print(f"  - {error['error_type']}: {error['details'][:100]}")

    # Save result
    output_file = folder / "merged.py"
    output_file.write_text(result.code, encoding="utf-8")
    print(f"\nResult saved to: {output_file}")

    return 0 if result.success else 1


def cmd_benchmark(args):
    """Run benchmark experiments."""
    # Validate API keys
    valid, missing = validate_api_keys()
    if not valid:
        print(f"Error: Missing API keys: {', '.join(missing)}")
        print("Set them in your .env file or environment.")
        return 1

    conflicts_dir = Path(args.conflicts) if args.conflicts else CONFLICTS_DIR
    output_dir = Path(args.output) if args.output else RESULTS_DIR

    if not conflicts_dir.exists():
        print(f"Error: Conflicts directory not found: {conflicts_dir}")
        return 1

    # Parse hypotheses
    hypotheses = None
    if args.hypothesis:
        if args.hypothesis.lower() == "all":
            hypotheses = ["H1", "H2", "H3"]
        else:
            hypotheses = [h.strip().upper() for h in args.hypothesis.split(",")]

    print("=" * 60)
    print("MERGE-AI BENCHMARK")
    print("=" * 60)

    if args.pilot:
        print("Running PILOT mode (5 conflicts, all hypotheses)")
        benchmark = Benchmark(conflicts_dir=conflicts_dir, output_dir=output_dir)
        result = benchmark.run_pilot(sample_size=5, verbose=not args.quiet)
    else:
        result = run_benchmark(
            conflicts_dir=conflicts_dir,
            output_dir=output_dir,
            hypotheses=hypotheses,
            sample_size=args.sample,
            max_cost_usd=args.max_cost,
            verbose=not args.quiet,
        )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    print(f"Total cost: ${result.total_cost_usd:.2f}")
    print(f"Results saved to: {output_dir}")

    return 0


def cmd_report(args):
    """Generate reports from experiment results."""
    results_dir = Path(args.results) if args.results else RESULTS_DIR

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return 1

    print(f"Generating reports from: {results_dir}")

    # Generate Markdown report
    if args.format in ["md", "all"]:
        output_path = Path(args.output) if args.output else results_dir / "FINDINGS.md"
        report = generate_report(results_dir, output_path)
        print(f"Markdown report: {output_path}")

    # Generate CSV summary
    if args.format in ["csv", "all"]:
        csv_path = results_dir / "summary.csv"
        generate_csv_summary(results_dir, csv_path)
        print(f"CSV summary: {csv_path}")

    # Generate error analysis
    if args.format in ["errors", "all"]:
        analyses = analyze_all_hypotheses(results_dir)
        if analyses:
            error_path = save_error_analysis(analyses, results_dir)
            print(f"Error analysis: {error_path}")
        else:
            print("No results found for error analysis")

    print("\nDone!")
    return 0


def cmd_list(args):
    """List available conflicts."""
    conflicts_dir = Path(args.conflicts) if args.conflicts else CONFLICTS_DIR

    if not conflicts_dir.exists():
        print(f"Conflicts directory not found: {conflicts_dir}")
        return 1

    conflicts = load_conflicts_from_dir(conflicts_dir)

    if not conflicts:
        print(f"No conflicts found in {conflicts_dir}")
        return 0

    print(f"Found {len(conflicts)} conflicts in {conflicts_dir}:\n")

    for i, conflict in enumerate(conflicts, 1):
        has_gold = "+" if conflict.gold else "-"
        base_lines = len(conflict.base.split("\n"))
        print(f"  {i:3}. {conflict.id:<30} ({base_lines:5} lines) [gold: {has_gold}]")

    return 0


def cmd_validate(args):
    """Validate experiment setup."""
    print("Validating MERGE-AI setup...\n")

    # Check API keys
    print("API Keys:")
    valid, missing = validate_api_keys()
    if valid:
        print("  [OK] All API keys configured")
    else:
        print(f"  [ERROR] Missing: {', '.join(missing)}")

    # Check conflicts directory
    conflicts_dir = Path(args.conflicts) if args.conflicts else CONFLICTS_DIR
    print(f"\nConflicts Directory: {conflicts_dir}")
    if conflicts_dir.exists():
        conflicts = load_conflicts_from_dir(conflicts_dir)
        print(f"  [OK] Found {len(conflicts)} conflicts")
        with_gold = sum(1 for c in conflicts if c.gold)
        print(f"  [OK] {with_gold} conflicts have gold standard")
    else:
        print("  [WARNING] Directory does not exist")

    # Check results directory
    results_dir = RESULTS_DIR
    print(f"\nResults Directory: {results_dir}")
    if results_dir.exists():
        json_files = list(results_dir.glob("*.json"))
        print(f"  [OK] Directory exists with {len(json_files)} result files")
    else:
        print("  [INFO] Directory does not exist (will be created)")

    print("\nValidation complete!")
    return 0 if valid else 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="merge_ai",
        description="MERGE-AI: Empirical Study of Constrained LLM Merge Resolution",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # resolve command
    resolve_parser = subparsers.add_parser(
        "resolve", help="Resolve a single merge conflict"
    )
    resolve_parser.add_argument(
        "--folder", "-f", required=True, help="Folder containing base.py, ours.py, theirs.py"
    )
    resolve_parser.add_argument(
        "--constraints",
        "-c",
        help="Constraints to apply (comma-separated: ast,imports,functions,all)",
    )
    resolve_parser.add_argument(
        "--max-retries", type=int, default=2, help="Maximum retry attempts (default: 2)"
    )

    # benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run benchmark experiments"
    )
    benchmark_parser.add_argument(
        "--conflicts", help="Path to conflicts directory"
    )
    benchmark_parser.add_argument(
        "--output", "-o", help="Path to output directory"
    )
    benchmark_parser.add_argument(
        "--hypothesis",
        help="Hypothesis to run: H1, H2, H3, or 'all' (default: all)",
    )
    benchmark_parser.add_argument(
        "--sample", type=int, default=30, help="Sample size per hypothesis (default: 30)"
    )
    benchmark_parser.add_argument(
        "--max-cost", type=float, default=100.0, help="Maximum cost in USD (default: 100)"
    )
    benchmark_parser.add_argument(
        "--pilot", action="store_true", help="Run pilot mode (5 conflicts)"
    )
    benchmark_parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress progress output"
    )

    # report command
    report_parser = subparsers.add_parser(
        "report", help="Generate reports from results"
    )
    report_parser.add_argument(
        "--results", "-r", help="Path to results directory"
    )
    report_parser.add_argument(
        "--output", "-o", help="Path to output file"
    )
    report_parser.add_argument(
        "--format",
        choices=["md", "csv", "errors", "all"],
        default="all",
        help="Report format (default: all)",
    )

    # list command
    list_parser = subparsers.add_parser("list", help="List available conflicts")
    list_parser.add_argument(
        "--conflicts", help="Path to conflicts directory"
    )

    # validate command
    validate_parser = subparsers.add_parser("validate", help="Validate setup")
    validate_parser.add_argument(
        "--conflicts", help="Path to conflicts directory"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "resolve": cmd_resolve,
        "benchmark": cmd_benchmark,
        "report": cmd_report,
        "list": cmd_list,
        "validate": cmd_validate,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
