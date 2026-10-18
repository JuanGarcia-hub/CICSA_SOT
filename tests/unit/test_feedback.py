"""Tests unitarios para dominio de feedback."""

import pytest
from sot_matcher.domain.feedback import (
    FeedbackDecision,
    FeedbackEvent,
    LabeledPair,
    CandidatePair,
    MatchingPrediction,
)


def test_feedback_decision_enum():
    assert FeedbackDecision.CORRECT == "CORRECT"
    assert FeedbackDecision.INCORRECT == "INCORRECT"
    assert FeedbackDecision.INCONCLUSIVE == "INCONCLUSIVE"


def test_feedback_event_training_validity():
    correct = FeedbackEvent(decision=FeedbackDecision.CORRECT)
    incorrect = FeedbackEvent(decision=FeedbackDecision.INCORRECT)
    inconclusive = FeedbackEvent(decision=FeedbackDecision.INCONCLUSIVE)

    assert correct.is_valid_for_training()
    assert incorrect.is_valid_for_training()
    assert not inconclusive.is_valid_for_training()


def test_feedback_event_label():
    correct = FeedbackEvent(decision=FeedbackDecision.CORRECT)
    incorrect = FeedbackEvent(decision=FeedbackDecision.INCORRECT)

    assert correct.label() == 1
    assert incorrect.label() == 0

    inconclusive = FeedbackEvent(decision=FeedbackDecision.INCONCLUSIVE)
    with pytest.raises(ValueError):
        inconclusive.label()


def test_feedback_event_supersedes():
    original = FeedbackEvent(feedback_id="orig", decision=FeedbackDecision.CORRECT)
    correction = FeedbackEvent(
        feedback_id="corr",
        decision=FeedbackDecision.INCORRECT,
        supersedes_feedback_id="orig",
    )
    assert correction.supersedes_feedback_id == "orig"


def test_feedback_event_immutable():
    event = FeedbackEvent(sot_id="SOT_1", log_id="LOG_1")
    with pytest.raises(Exception):
        event.sot_id = "SOT_2"  # type: ignore


def test_labeled_pair_creation():
    pair = LabeledPair(
        sot_id="SOT_1",
        log_id="LOG_1",
        label=1,
        features={"feat1": 0.5},
        feature_schema_version="1.0",
        feedback_ids=("fb_1",),
    )
    assert pair.label == 1
    assert pair.features["feat1"] == 0.5


def test_candidate_pair_defaults():
    pair = CandidatePair(sot_id="SOT_1", log_id="LOG_1")
    assert pair.feature_schema_version == "1.0"
    assert pair.features == {}
    assert pair.semantic_score is None


def test_matching_prediction_defaults():
    pred = MatchingPrediction(sot_id="SOT_1", log_id="LOG_1")
    assert pred.recommended_decision == FeedbackDecision.INCONCLUSIVE
    assert pred.evidence == {}