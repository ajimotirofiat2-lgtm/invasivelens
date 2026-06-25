"""
Decision support engine for invasive species management.

Transforms model predictions into actionable recommendations for:
- Risk assessment
- Habitat management decisions
- Field survey prioritization
- Resource allocation

Usage:
    from src.decision_engine import InvasiveSpeciesDecisionEngine
    
    engine = InvasiveSpeciesDecisionEngine(model_name="resnet50", threshold=0.7)
    decision = engine.assess_location("image.jpg", lat=40.2, lon=-8.5)
    print(decision.recommendation)  # "HIGH_PRIORITY: Arundo donax detected with 0.92 confidence"
"""
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from config import CANDIDATE_PAIRS
from src.inference import InvasiveLensPredictor, Prediction


class RiskLevel(Enum):
    """Risk assessment levels."""
    CRITICAL = "CRITICAL"      # Confirmed invasive, immediate action needed
    HIGH = "HIGH"              # Likely invasive, urgent monitoring
    MEDIUM = "MEDIUM"          # Possible invasive, requires verification
    LOW = "LOW"                # Unlikely invasive, routine monitoring
    NATIVE = "NATIVE"          # Native species, no action needed


class ActionType(Enum):
    """Recommended management actions."""
    IMMEDIATE_REMOVAL = "Immediate removal"
    HERBICIDE_TREATMENT = "Herbicide treatment"
    MECHANICAL_REMOVAL = "Mechanical removal"
    CONTAINMENT = "Containment measures"
    MONITORING = "Monitoring"
    NO_ACTION = "No action"


@dataclass
class DecisionResult:
    """Result of species/risk assessment decision."""
    species: str
    confidence: float
    risk_level: RiskLevel
    recommended_action: ActionType
    reasoning: str
    follow_up_actions: List[str]

    def __str__(self) -> str:
        return (
            f"{self.risk_level.value} RISK: {self.species} "
            f"(confidence: {self.confidence:.1%})\n"
            f"Action: {self.recommended_action.value}\n"
            f"Reasoning: {self.reasoning}"
        )


class InvasiveSpeciesDecisionEngine:
    """
    Decision engine for invasive species management.

    Converts model predictions into risk assessments and recommendations.
    """

    def __init__(
        self,
        model_name: str = "resnet50",
        fold: int = 0,
        confidence_threshold: float = 0.7,
        high_confidence_threshold: float = 0.9,
    ):
        """
        Args:
            model_name: "baseline", "resnet50", or "efficientnet_v2_s"
            fold: Model fold (0-4)
            confidence_threshold: Minimum confidence to make a decision
            high_confidence_threshold: Confidence level considered "high"
        """
        self.predictor = InvasiveLensPredictor(
            model_name=model_name, fold=fold, threshold=confidence_threshold
        )
        self.confidence_threshold = confidence_threshold
        self.high_confidence_threshold = high_confidence_threshold

        # Build species info lookup
        self.species_info = {}
        for pair in CANDIDATE_PAIRS:
            invasive = pair["invasive"]
            native = pair.get("native", "Unknown")
            self.species_info[invasive.lower()] = {
                "invasive_name": invasive,
                "native_comparison": native,
                "priority": pair.get("priority", "normal"),
                "impacts": pair.get("description", ""),
            }

    def assess_prediction(self, pred: Prediction) -> DecisionResult:
        """
        Convert a model prediction to a decision.

        Args:
            pred: Prediction from InvasiveLensPredictor

        Returns:
            DecisionResult with risk level and recommended action
        """
        species = pred.predicted_class
        confidence = pred.confidence
        species_lower = species.lower()

        # Check if species is invasive
        is_invasive = any(sp.lower() == species_lower for sp, _ in self.species_info.items())

        if pred.abstain:
            # Low confidence - require verification
            return DecisionResult(
                species=species,
                confidence=confidence,
                risk_level=RiskLevel.MEDIUM,
                recommended_action=ActionType.MONITORING,
                reasoning=f"Low confidence detection ({confidence:.1%}). Requires field verification.",
                follow_up_actions=[
                    "Manual photo review recommended",
                    "Collect multiple angle shots",
                    "Document GPS location precisely",
                    "Re-photograph in different lighting",
                ],
            )

        if not is_invasive:
            return DecisionResult(
                species=species,
                confidence=confidence,
                risk_level=RiskLevel.NATIVE,
                recommended_action=ActionType.NO_ACTION,
                reasoning=f"Native or non-target species detected ({confidence:.1%} confidence).",
                follow_up_actions=["Document sighting for biodiversity records"],
            )

        # Invasive species - assess risk based on confidence
        if confidence >= self.high_confidence_threshold:
            risk_level = RiskLevel.CRITICAL
            action = ActionType.IMMEDIATE_REMOVAL
            reasoning = f"High-confidence invasive species detection ({confidence:.1%})."
            follow_up = [
                "Schedule immediate removal assessment",
                "Check for ecological damage indicators",
                "Document population size and extent",
                "Notify local environmental authorities",
                "Plan management strategy",
            ]
        elif confidence >= self.confidence_threshold:
            risk_level = RiskLevel.HIGH
            action = ActionType.CONTAINMENT
            reasoning = f"Confirmed invasive species detection ({confidence:.1%})."
            follow_up = [
                "Initiate containment protocol",
                "Schedule removal within 30 days",
                "Mark area for monitoring",
                "Check for spread to adjacent areas",
                "Plan multi-year management if extensive",
            ]
        else:
            # Shouldn't reach here due to abstention, but handle gracefully
            risk_level = RiskLevel.MEDIUM
            action = ActionType.MONITORING
            reasoning = f"Possible invasive species ({confidence:.1%}). Requires verification."
            follow_up = [
                "Schedule field verification",
                "Collect reference samples",
                "Document location and habitat",
            ]

        # Get species-specific information
        species_data = self.species_info.get(species_lower, {})

        return DecisionResult(
            species=species,
            confidence=confidence,
            risk_level=risk_level,
            recommended_action=action,
            reasoning=reasoning,
            follow_up_actions=follow_up,
        )

    def assess_location(
        self,
        image_path: str | Path,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
    ) -> DecisionResult:
        """
        Assess a location photo for invasive species.

        Args:
            image_path: Path to image file
            latitude: GPS latitude (optional, for context)
            longitude: GPS longitude (optional, for context)

        Returns:
            DecisionResult with recommendation
        """
        pred = self.predictor.predict_from_file(image_path)
        return self.assess_prediction(pred)

    def assess_batch(
        self, image_paths: List[str | Path], export_csv: Optional[Path] = None
    ) -> List[DecisionResult]:
        """
        Assess multiple locations (e.g., field survey batch).

        Args:
            image_paths: List of image file paths
            export_csv: Optional path to export results as CSV

        Returns:
            List of DecisionResult objects
        """
        results = []
        preds = self.predictor.predict_batch(image_paths)

        for pred in preds:
            decision = self.assess_prediction(pred)
            results.append(decision)

        # Export if requested
        if export_csv:
            self._export_batch_results(results, export_csv)

        return results

    def _export_batch_results(self, results: List[DecisionResult], csv_path: Path):
        """Export batch assessment results to CSV."""
        import pandas as pd

        rows = []
        for result in results:
            rows.append({
                "species": result.species,
                "confidence": f"{result.confidence:.2%}",
                "risk_level": result.risk_level.value,
                "recommended_action": result.recommended_action.value,
                "reasoning": result.reasoning,
                "follow_up_count": len(result.follow_up_actions),
            })

        df = pd.DataFrame(rows)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"[OK] Exported {len(rows)} assessments to {csv_path}")

    def generate_report(self, results: List[DecisionResult]) -> str:
        """
        Generate a human-readable assessment report.

        Useful for field managers and decision makers.
        """
        report = []
        report.append("=" * 70)
        report.append("INVASIVE SPECIES ASSESSMENT REPORT")
        report.append("=" * 70)

        # Summary statistics
        by_risk = {}
        for r in results:
            risk = r.risk_level.value
            by_risk[risk] = by_risk.get(risk, 0) + 1

        report.append("\nSUMMARY:")
        for risk, count in sorted(by_risk.items()):
            report.append(f"  {risk}: {count}")

        # Critical findings
        critical = [r for r in results if r.risk_level == RiskLevel.CRITICAL]
        if critical:
            report.append("\nCRITICAL FINDINGS (IMMEDIATE ACTION REQUIRED):")
            for r in critical:
                report.append(f"\n  {r.species} ({r.confidence:.1%})")
                report.append(f"  Action: {r.recommended_action.value}")
                report.append(f"  Reasoning: {r.reasoning}")
                if r.follow_up_actions:
                    report.append("  Next steps:")
                    for action in r.follow_up_actions:
                        report.append(f"    • {action}")

        # High priority findings
        high = [r for r in results if r.risk_level == RiskLevel.HIGH]
        if high:
            report.append("\nHIGH PRIORITY FINDINGS:")
            for r in high:
                report.append(f"\n  {r.species} ({r.confidence:.1%})")
                report.append(f"  Action: {r.recommended_action.value}")

        # Recommendations
        report.append("\n" + "=" * 70)
        report.append("RECOMMENDED ACTIONS:")
        report.append("=" * 70)

        actions_needed = {}
        for r in results:
            if r.risk_level in [RiskLevel.CRITICAL, RiskLevel.HIGH]:
                action = r.recommended_action.value
                actions_needed[action] = actions_needed.get(action, 0) + 1

        for action, count in sorted(actions_needed.items()):
            report.append(f"  • {action}: {count} locations")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    engine = InvasiveSpeciesDecisionEngine(model_name="resnet50", confidence_threshold=0.7)
    print("Decision Engine initialized. Ready for use.")
