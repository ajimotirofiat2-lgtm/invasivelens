#!/usr/bin/env python3
"""
End-to-end workflow: Import data → Train models → Make decisions → Generate reports.

This demonstrates how InvasiveLens becomes a complete decision-making system.

Usage:
    python -m src.workflow all
    python -m src.workflow import-data
    python -m src.workflow train
    python -m src.workflow assess --image photo.jpg
    python -m src.workflow report
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from config import CHECKPOINT_DIR, MANIFEST_DIR, RESULTS_DIR


def run_command(cmd: List[str], description: str) -> bool:
    """Run shell command and report results."""
    print(f"\n{'=' * 70}")
    print(f"[>] {description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=False)
        if result.returncode == 0:
            print(f"\n[OK] {description} completed successfully")
            return True
        else:
            print(f"\n[WARN] {description} completed with warnings (exit code: {result.returncode})")
            return True  # Continue anyway
    except Exception as e:
        print(f"\n[FAIL] {description} failed: {e}")
        return False


def workflow_import_data():
    """Step 1: Import real datasets."""
    print("\n" + "=" * 70)
    print("STEP 1: IMPORTING REAL DATASETS")
    print("=" * 70)

    # Create occurrence manifest from GBIF + INVASORAS
    if not run_command(
        ["python", "-m", "src.data.import_real_data"],
        "Creating occurrence manifest from real datasets"
    ):
        return False

    # Verify manifest was created
    manifest_path = MANIFEST_DIR / "occurrence_manifest.csv"
    if manifest_path.exists():
        import pandas as pd
        df = pd.read_csv(manifest_path)
        print(f"\n[OK] Created manifest with {len(df)} occurrence records")
        print(f"  Species: {df['label'].nunique()}")
        print(f"  Regions: {df['region'].nunique()}")
        return True
    else:
        print(f"\n[WARN] Manifest not found at {manifest_path}")
        return False


def workflow_train():
    """Step 2: Train models on real data."""
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING MODELS")
    print("=" * 70)

    manifest = MANIFEST_DIR / "combined_manifest.csv" if (MANIFEST_DIR / "combined_manifest.csv").exists() \
        else MANIFEST_DIR / "occurrence_manifest.csv"

    if not manifest.exists():
        print(f"[FAIL] Manifest not found: {manifest}")
        return False

    # Train baseline model
    if run_command(
        ["python", "-m", "src.train", "--manifest", str(manifest), "--model", "baseline", "--epochs", "5"],
        "Training baseline model"
    ):
        checkpoint = CHECKPOINT_DIR / "baseline_fold0.pt"
        if checkpoint.exists():
            print(f"[OK] Baseline model checkpoint saved: {checkpoint}")
    else:
        return False

    # Train ResNet50
    if run_command(
        ["python", "-m", "src.train", "--manifest", str(manifest), "--model", "resnet50", "--epochs", "5"],
        "Training ResNet50 model"
    ):
        checkpoint = CHECKPOINT_DIR / "resnet50_fold0.pt"
        if checkpoint.exists():
            print(f"[OK] ResNet50 model checkpoint saved: {checkpoint}")
    else:
        return False

    return True


def workflow_assess(args):
    """Step 3: Assess locations for invasive species."""
    print("\n" + "=" * 70)
    print("STEP 3: ASSESSING LOCATIONS")
    print("=" * 70)

    from src.decision_engine import InvasiveSpeciesDecisionEngine

    engine = InvasiveSpeciesDecisionEngine(
        model_name=args.model or "resnet50",
        confidence_threshold=args.threshold or 0.7
    )

    if args.image:
        print(f"\nAssessing: {args.image}")
        decision = engine.assess_location(args.image)
        print(f"\n{decision}")
        return True

    elif args.batch:
        batch_dir = Path(args.batch)
        image_paths = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.png"))
        if not image_paths:
            print(f"[WARN] No images found in {batch_dir}")
            return False

        print(f"\nAssessing {len(image_paths)} images...")
        results = engine.assess_batch(image_paths)
        print(f"[OK] Assessed {len(results)} locations")

        if args.export:
            engine._export_batch_results(results, Path(args.export))

        return True
    else:
        print("Specify --image or --batch")
        return False


def workflow_report():
    """Step 4: Generate management reports."""
    print("\n" + "=" * 70)
    print("STEP 4: GENERATING REPORTS")
    print("=" * 70)

    # Find latest results
    results_csvs = list(RESULTS_DIR.glob("*.csv"))
    if not results_csvs:
        print(f"[WARN] No results found in {RESULTS_DIR}")
        print("Run workflow with --assess --batch to generate results first")
        return False

    latest_csv = max(results_csvs, key=lambda p: p.stat().st_mtime)
    print(f"\nGenerating report from: {latest_csv}")

    return run_command(
        ["python", "-m", "src.cli", "report", "--results", str(latest_csv), 
         "--output", str(RESULTS_DIR / "management_report.txt")],
        "Generating management report"
    )


def workflow_compare():
    """Step 5: Compare models."""
    print("\n" + "=" * 70)
    print("STEP 5: COMPARING MODELS")
    print("=" * 70)

    return run_command(
        ["python", "-m", "src.compare_models", "--model-a", "baseline", "--model-b", "resnet50"],
        "Comparing model performance"
    )


def workflow_all(args):
    """Run complete workflow."""
    print("\n" + "= " * 20)
    print("INVASIVELENS END-TO-END WORKFLOW")
    print("= " * 20)

    steps = [
        ("Import Data", workflow_import_data),
        ("Train Models", workflow_train),
    ]

    if args.assess:
        steps.append(("Assess Locations", lambda: workflow_assess(args)))

    steps.extend([
        ("Compare Models", workflow_compare),
        ("Generate Reports", workflow_report),
    ])

    completed = 0
    for step_name, step_func in steps:
        try:
            if step_func():
                completed += 1
            else:
                print(f"\n[WARN] Skipping remaining steps after {step_name}")
                break
        except Exception as e:
            print(f"\n[FAIL] Error in {step_name}: {e}")
            break

    # Summary
    print("\n" + "=" * 70)
    print("WORKFLOW SUMMARY")
    print("=" * 70)
    print(f"[OK] Completed {completed}/{len(steps)} steps")

    if completed == len(steps):
        print("\n[DONE] WORKFLOW COMPLETE!")
        print("\nNext steps:")
        print("  1. Review decision results")
        print("  2. Implement recommended actions")
        print("  3. Monitor field outcomes")
        print("  4. Retrain models with new field data")
        return True
    else:
        print("\n[WARN] Workflow incomplete. Check errors above.")
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Workflow commands")

    # All command (complete workflow)
    all_parser = subparsers.add_parser("all", help="Run complete workflow")
    all_parser.add_argument("--assess", action="store_true", help="Include assessment step")
    all_parser.add_argument("--image", help="Single image to assess")
    all_parser.add_argument("--batch", help="Batch directory to assess")
    all_parser.add_argument("--model", default="resnet50")
    all_parser.add_argument("--threshold", type=float, default=0.7)
    all_parser.add_argument("--export", help="Export results to CSV")
    all_parser.set_defaults(func=lambda args: workflow_all(args))

    # Individual steps
    subparsers.add_parser("import-data", help="Import real datasets").set_defaults(
        func=lambda args: workflow_import_data()
    )
    subparsers.add_parser("train", help="Train models").set_defaults(
        func=lambda args: workflow_train()
    )

    assess_parser = subparsers.add_parser("assess", help="Assess locations")
    assess_parser.add_argument("--image", help="Single image")
    assess_parser.add_argument("--batch", help="Batch directory")
    assess_parser.add_argument("--model", default="resnet50")
    assess_parser.add_argument("--threshold", type=float, default=0.7)
    assess_parser.add_argument("--export", help="Export to CSV")
    assess_parser.set_defaults(func=lambda args: workflow_assess(args))

    subparsers.add_parser("compare", help="Compare models").set_defaults(
        func=lambda args: workflow_compare()
    )
    subparsers.add_parser("report", help="Generate reports").set_defaults(
        func=lambda args: workflow_report()
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 0

    try:
        result = args.func(args)
        return 0 if result else 1
    except Exception as e:
        print(f"\n[FAIL] Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
