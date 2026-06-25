"""
Tests for src/decision_engine.py — the risk assignment and management
recommendation logic.

This is the safety-critical component: incorrect risk levels could mean
removing native plants (false positive) or ignoring invasive ones (false
negative). Every branch in assess_prediction is tested against known
inputs to verify the decision table is implemented correctly.

Run with:  pytest tests/test_decision_engine.py -v
"""
import pytest

from src.decision_engine import (
    ActionType,
    DecisionResult,
    InvasiveSpeciesDecisionEngine,
    RiskLevel,
)
from src.inference import Prediction


# ---------------------------------------------------------------------------
# Helpers — build Prediction objects without needing a real model
# ---------------------------------------------------------------------------

def _make_prediction(species: str, confidence: float, threshold: float = 0.7) -> Prediction:
    """Build a Prediction with the given species/confidence.
    
    abstain is True when confidence < threshold, matching InvasiveLensPredictor
    behaviour.
    """
    return Prediction(
        predicted_class=species,
        confidence=confidence,
        abstain=confidence < threshold,
        top_3_classes=[(species, confidence), ("other_a", 0.05), ("other_b", 0.02)],
    )


@pytest.fixture
def engine():
    """Decision engine with default thresholds (70% decision, 90% critical).
    
    We can't load a real model checkpoint in unit tests, so we only test
    assess_prediction() which takes a Prediction object directly — the
    assess_location() path (which calls the predictor) is an integration
    test concern.
    """
    # Build engine without loading a model by mocking the predictor init.
    # We only need the species_info dict and threshold attributes.
    eng = object.__new__(InvasiveSpeciesDecisionEngine)
    eng.confidence_threshold = 0.7
    eng.high_confidence_threshold = 0.9

    # Manually populate species_info from CANDIDATE_PAIRS
    from config import CANDIDATE_PAIRS
    eng.species_info = {}
    for pair in CANDIDATE_PAIRS:
        invasive = pair["invasive"]
        native = pair.get("native", "Unknown")
        eng.species_info[invasive.lower()] = {
            "invasive_name": invasive,
            "native_comparison": native,
            "priority": pair.get("priority", "normal"),
            "impacts": pair.get("description", ""),
        }

    return eng


# ---------------------------------------------------------------------------
# Core decision table tests
# ---------------------------------------------------------------------------

class TestRiskLevelAssignment:
    """Verify every branch of the decision logic."""

    def test_low_confidence_triggers_abstain_medium_risk(self, engine):
        """Confidence < 70% → MEDIUM risk, regardless of species."""
        pred = _make_prediction("Arundo donax", confidence=0.55)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.recommended_action == ActionType.MONITORING
        assert "verification" in result.reasoning.lower() or "low confidence" in result.reasoning.lower()

    def test_high_confidence_invasive_is_critical(self, engine):
        """Invasive species at ≥90% → CRITICAL, immediate removal."""
        pred = _make_prediction("Arundo donax", confidence=0.95)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.CRITICAL
        assert result.recommended_action == ActionType.IMMEDIATE_REMOVAL

    def test_medium_confidence_invasive_is_high(self, engine):
        """Invasive species at 70–90% → HIGH, containment."""
        pred = _make_prediction("Cortaderia selloana", confidence=0.78)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.HIGH
        assert result.recommended_action == ActionType.CONTAINMENT

    def test_native_species_is_native_risk(self, engine):
        """Non-invasive species at ≥70% → NATIVE, no action."""
        pred = _make_prediction("Phragmites australis", confidence=0.85)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.NATIVE
        assert result.recommended_action == ActionType.NO_ACTION

    def test_boundary_at_exactly_70_percent(self, engine):
        """Exactly 70% confidence on invasive → should be HIGH (not MEDIUM)."""
        pred = _make_prediction("Arundo donax", confidence=0.70, threshold=0.70)
        result = engine.assess_prediction(pred)

        # 70% is at the threshold, so abstain=False, and 70% < 90% → HIGH
        assert result.risk_level == RiskLevel.HIGH

    def test_boundary_at_exactly_90_percent(self, engine):
        """Exactly 90% confidence on invasive → should be CRITICAL."""
        pred = _make_prediction("Acacia dealbata", confidence=0.90)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.CRITICAL

    def test_just_below_70_percent_abstains(self, engine):
        """69.9% confidence → abstain, MEDIUM risk."""
        pred = _make_prediction("Arundo donax", confidence=0.699)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.MEDIUM
        assert result.recommended_action == ActionType.MONITORING

    def test_just_below_90_percent_is_high_not_critical(self, engine):
        """89.9% confidence on invasive → HIGH, not CRITICAL."""
        pred = _make_prediction("Cortaderia selloana", confidence=0.899)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.HIGH
        assert result.recommended_action == ActionType.CONTAINMENT


# ---------------------------------------------------------------------------
# Species recognition tests
# ---------------------------------------------------------------------------

class TestSpeciesRecognition:
    """Verify invasive/non-invasive classification matches config."""

    def test_all_invasive_species_are_recognized(self, engine):
        """Every invasive species from CANDIDATE_PAIRS should trigger risk."""
        from config import CANDIDATE_PAIRS

        for pair in CANDIDATE_PAIRS:
            invasive = pair["invasive"]
            pred = _make_prediction(invasive, confidence=0.95)
            result = engine.assess_prediction(pred)

            assert result.risk_level == RiskLevel.CRITICAL, (
                f"{invasive} at 95% confidence should be CRITICAL, got {result.risk_level}"
            )

    def test_native_counterparts_are_not_invasive(self, engine):
        """Native counterparts should get NATIVE risk level."""
        from config import CANDIDATE_PAIRS

        for pair in CANDIDATE_PAIRS:
            native = pair["native"]
            pred = _make_prediction(native, confidence=0.85)
            result = engine.assess_prediction(pred)

            assert result.risk_level == RiskLevel.NATIVE, (
                f"{native} at 85% confidence should be NATIVE, got {result.risk_level}"
            )

    def test_unknown_species_treated_as_native(self, engine):
        """A species not in CANDIDATE_PAIRS is treated as non-invasive."""
        pred = _make_prediction("Quercus robur", confidence=0.90)
        result = engine.assess_prediction(pred)

        assert result.risk_level == RiskLevel.NATIVE

    def test_species_lookup_is_case_insensitive(self, engine):
        """'arundo donax' and 'Arundo donax' should both be recognised."""
        pred_lower = _make_prediction("arundo donax", confidence=0.92)
        pred_title = _make_prediction("Arundo donax", confidence=0.92)

        result_lower = engine.assess_prediction(pred_lower)
        result_title = engine.assess_prediction(pred_title)

        assert result_lower.risk_level == result_title.risk_level == RiskLevel.CRITICAL


# ---------------------------------------------------------------------------
# Follow-up actions tests
# ---------------------------------------------------------------------------

class TestFollowUpActions:
    """Verify that follow-up actions are populated and appropriate."""

    def test_critical_has_multiple_follow_ups(self, engine):
        pred = _make_prediction("Arundo donax", confidence=0.95)
        result = engine.assess_prediction(pred)

        assert len(result.follow_up_actions) >= 3
        actions_text = " ".join(result.follow_up_actions).lower()
        assert "removal" in actions_text or "authorities" in actions_text

    def test_medium_risk_recommends_re_photographing(self, engine):
        pred = _make_prediction("Arundo donax", confidence=0.55)
        result = engine.assess_prediction(pred)

        assert len(result.follow_up_actions) >= 2
        actions_text = " ".join(result.follow_up_actions).lower()
        assert "review" in actions_text or "photo" in actions_text

    def test_native_has_minimal_follow_ups(self, engine):
        pred = _make_prediction("Phragmites australis", confidence=0.85)
        result = engine.assess_prediction(pred)

        assert len(result.follow_up_actions) >= 1
        assert len(result.follow_up_actions) <= 3  # shouldn't be a long action list


# ---------------------------------------------------------------------------
# DecisionResult formatting tests
# ---------------------------------------------------------------------------

class TestDecisionResultFormatting:

    def test_str_contains_risk_and_species(self, engine):
        pred = _make_prediction("Arundo donax", confidence=0.95)
        result = engine.assess_prediction(pred)
        text = str(result)

        assert "CRITICAL" in text
        assert "Arundo donax" in text
        assert "Immediate removal" in text

    def test_str_contains_confidence_as_percentage(self, engine):
        pred = _make_prediction("Arundo donax", confidence=0.873)
        result = engine.assess_prediction(pred)
        text = str(result)

        assert "87.3%" in text


# ---------------------------------------------------------------------------
# Batch assessment tests
# ---------------------------------------------------------------------------

class TestBatchAssessment:

    def test_assess_batch_returns_correct_count(self, engine):
        """assess_prediction on multiple inputs returns one result per input."""
        preds = [
            _make_prediction("Arundo donax", confidence=0.95),
            _make_prediction("Phragmites australis", confidence=0.80),
            _make_prediction("Cortaderia selloana", confidence=0.55),
        ]
        results = [engine.assess_prediction(p) for p in preds]

        assert len(results) == 3
        assert results[0].risk_level == RiskLevel.CRITICAL
        assert results[1].risk_level == RiskLevel.NATIVE
        assert results[2].risk_level == RiskLevel.MEDIUM


# ---------------------------------------------------------------------------
# Report generation tests
# ---------------------------------------------------------------------------

class TestReportGeneration:

    def test_report_contains_summary_and_actions(self, engine):
        preds = [
            _make_prediction("Arundo donax", confidence=0.95),
            _make_prediction("Cortaderia selloana", confidence=0.78),
            _make_prediction("Phragmites australis", confidence=0.85),
        ]
        results = [engine.assess_prediction(p) for p in preds]
        report = engine.generate_report(results)

        assert "ASSESSMENT REPORT" in report
        assert "CRITICAL" in report
        assert "HIGH" in report
        assert "RECOMMENDED ACTIONS" in report

    def test_report_with_no_critical_findings(self, engine):
        preds = [
            _make_prediction("Phragmites australis", confidence=0.85),
            _make_prediction("Ammophila arenaria", confidence=0.90),
        ]
        results = [engine.assess_prediction(p) for p in preds]
        report = engine.generate_report(results)

        assert "CRITICAL FINDINGS" not in report

    def test_empty_results_produce_valid_report(self, engine):
        report = engine.generate_report([])

        assert "ASSESSMENT REPORT" in report
        assert isinstance(report, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
