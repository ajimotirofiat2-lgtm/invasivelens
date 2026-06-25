#!/usr/bin/env python3
"""
Command-line tool for field surveyors to identify and assess invasive plants.

This tool turns model predictions into practical management decisions for
environmental monitoring teams.

Examples:
    # Single photo assessment
    python -m src.cli survey --image photo.jpg --model resnet50 --threshold 0.7
    
    # Batch field survey (process multiple photos)
    python -m src.cli survey --batch survey_photos/ --model resnet50 --export results.csv
    
    # Real-time survey with GPS integration
    python -m src.cli survey --gps --timeout 60
    
    # Risk map for region
    python -m src.cli risk-map --manifest manifests/occurrence_manifest.csv
    
    # Generate management recommendations
    python -m src.cli report --results results.csv --output report.txt
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from src.decision_engine import InvasiveSpeciesDecisionEngine, RiskLevel
from src.inference import InvasiveLensPredictor


def cmd_survey(args):
    """Assess images for invasive species."""
    engine = InvasiveSpeciesDecisionEngine(
        model_name=args.model,
        fold=args.fold,
        confidence_threshold=args.threshold,
    )

    if args.image:
        # Single image
        print(f"\nAnalyzing: {args.image}")
        decision = engine.assess_location(args.image)
        print(f"\n{decision}")

        if args.save:
            Path(args.save).write_text(str(decision))
            print(f"\n[OK] Decision saved to {args.save}")

    elif args.batch:
        # Batch processing
        batch_dir = Path(args.batch)
        image_paths = list(batch_dir.glob("*.jpg")) + list(batch_dir.glob("*.JPG")) + \
                      list(batch_dir.glob("*.png")) + list(batch_dir.glob("*.PNG"))

        if not image_paths:
            print(f"[FAIL] No images found in {batch_dir}")
            return

        print(f"\nProcessing {len(image_paths)} images from {batch_dir}")
        results = engine.assess_batch(image_paths)

        # Print summary
        critical = len([r for r in results if r.risk_level == RiskLevel.CRITICAL])
        high = len([r for r in results if r.risk_level == RiskLevel.HIGH])
        
        print(f"\nSummary:")
        print(f"  CRITICAL: {critical}")
        print(f"  HIGH: {high}")
        print(f"  NATIVE: {len([r for r in results if r.risk_level == RiskLevel.NATIVE])}")

        # Export if requested
        if args.export:
            engine._export_batch_results(results, Path(args.export))
            # Also generate report
            report = engine.generate_report(results)
            print(f"\n{report}")

    elif args.gps:
        print("GPS mode: Experimental feature")
        print("(Requires GPS-enabled device or manual lat/lon input)")


def cmd_identify(args):
    """Quickly identify a species from a photo."""
    predictor = InvasiveLensPredictor(
        model_name=args.model,
        fold=args.fold,
        threshold=args.threshold,
    )

    pred = predictor.predict_from_file(args.image)

    print(f"\nSpecies Identification")
    print(f"  Predicted: {pred.predicted_class}")
    print(f"  Confidence: {pred.confidence:.1%}")
    print(f"  Top 3: {pred.top_3_classes}")

    if pred.abstain:
        print(f"  [WARN] Confidence too low for decision. Manual review recommended.")


def cmd_riskmap(args):
    """Generate risk map for a region."""
    print("\nGenerating Risk Map")
    print("(Requires occurrence manifest with latitude/longitude data)")

    import pandas as pd

    manifest_df = pd.read_csv(args.manifest)

    # Simple spatial binning (10x10 grid)
    print(f"\nAnalyzing {len(manifest_df)} occurrence records...")

    # Count invasive occurrences by region
    if "region" in manifest_df.columns:
        region_risk = manifest_df[manifest_df["status"] == "invasive"]["region"].value_counts()
        print("\nInvasive occurrences by region:")
        for region, count in region_risk.head(10).items():
            risk_score = count / len(manifest_df[manifest_df["region"] == region])
            print(f"  {region}: {count} occurrences (risk score: {risk_score:.1%})")

    # Count by species
    if "label" in manifest_df.columns:
        species_risk = manifest_df[manifest_df["status"] == "invasive"]["label"].value_counts()
        print("\nInvasive occurrences by species:")
        for species, count in species_risk.items():
            print(f"  {species}: {count}")


def cmd_calibration(args):
    """Show model calibration info."""
    predictor = InvasiveLensPredictor(model_name=args.model, fold=args.fold)
    calib = predictor.get_calibration_info()

    print(f"\nModel Calibration ({args.model}, fold {args.fold})")
    print(f"  Accuracy on fold-0: {calib.get('accuracy', 'N/A'):.1%}")
    print(f"  Mean confidence (correct): {calib.get('mean_confidence_correct', 'N/A'):.1%}")
    print(f"  Mean confidence (incorrect): {calib.get('mean_confidence_incorrect', 'N/A'):.1%}")
    print(f"  Confidence range: {calib.get('min_confidence', 'N/A'):.1%} - {calib.get('max_confidence', 'N/A'):.1%}")


def cmd_report(args):
    """Generate management report."""
    print("\nGenerating Management Report")

    if args.results.endswith(".csv"):
        import pandas as pd
        results_df = pd.read_csv(args.results)

        report = []
        report.append("=" * 70)
        report.append("INVASIVE SPECIES MANAGEMENT REPORT")
        report.append("=" * 70)
        report.append(f"\nGenerated from: {args.results}")
        report.append(f"Total assessments: {len(results_df)}")

        # Summary by risk level
        if "risk_level" in results_df.columns:
            report.append("\n--- RISK SUMMARY ---")
            for risk, count in results_df["risk_level"].value_counts().items():
                report.append(f"  {risk}: {count}")

        # Recommended actions
        if "recommended_action" in results_df.columns:
            report.append("\n--- RECOMMENDED ACTIONS ---")
            for action, count in results_df["recommended_action"].value_counts().items():
                report.append(f"  {action}: {count} locations")

        report_text = "\n".join(report)
        print(f"\n{report_text}")

        if args.output:
            Path(args.output).write_text(report_text)
            print(f"\n[OK] Report saved to {args.output}")
    else:
        print("[FAIL] Unsupported format. Requires .csv results file.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Survey command
    survey_parser = subparsers.add_parser("survey", help="Assess images for invasive species")
    survey_parser.add_argument("--image", help="Single image to assess")
    survey_parser.add_argument("--batch", help="Directory of images to batch process")
    survey_parser.add_argument("--gps", action="store_true",
                              help="Enable GPS-based field survey mode")
    survey_parser.add_argument("--model", default="resnet50",
                              choices=["baseline", "resnet50", "efficientnet_v2_s"])
    survey_parser.add_argument("--fold", type=int, default=0)
    survey_parser.add_argument("--threshold", type=float, default=0.7,
                              help="Confidence threshold for decisions")
    survey_parser.add_argument("--export", help="Export batch results to CSV")
    survey_parser.add_argument("--save", help="Save decision to text file")
    survey_parser.set_defaults(func=cmd_survey)

    # Identify command
    identify_parser = subparsers.add_parser("identify", help="Quick species identification")
    identify_parser.add_argument("image", help="Image to identify")
    identify_parser.add_argument("--model", default="resnet50",
                                choices=["baseline", "resnet50", "efficientnet_v2_s"])
    identify_parser.add_argument("--fold", type=int, default=0)
    identify_parser.add_argument("--threshold", type=float, default=0.5)
    identify_parser.set_defaults(func=cmd_identify)

    # Risk map command
    riskmap_parser = subparsers.add_parser("risk-map", help="Generate risk map")
    riskmap_parser.add_argument("--manifest", required=True,
                               help="Manifest with occurrence data")
    riskmap_parser.set_defaults(func=cmd_riskmap)

    # Calibration command
    calib_parser = subparsers.add_parser("calibration", help="Show model calibration")
    calib_parser.add_argument("--model", default="resnet50",
                             choices=["baseline", "resnet50", "efficientnet_v2_s"])
    calib_parser.add_argument("--fold", type=int, default=0)
    calib_parser.set_defaults(func=cmd_calibration)

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate management report")
    report_parser.add_argument("--results", required=True, help="Results CSV file")
    report_parser.add_argument("--output", help="Output report file")
    report_parser.set_defaults(func=cmd_report)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    try:
        args.func(args)
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
