"""Tests unitarios para CatBoostMatchingClassifier."""

import pytest
import numpy as np
from sot_matcher.infrastructure.classifier import CatBoostMatchingClassifier
from sot_matcher.domain.feedback import LabeledPair


@pytest.fixture
def classifier():
    return CatBoostMatchingClassifier(iterations=10, verbose=False, random_seed=42)


@pytest.fixture
def sample_labeled_pairs():
    """Genera pares etiquetados sintéticos para testing."""
    pairs = []
    # 5 positivos
    for i in range(5):
        pairs.append(LabeledPair(
            sot_id=f"SOT_{i}",
            log_id=f"LOG_{i}",
            label=1,
            features={
                "incidencia_exacta": 1.0,
                "sot_en_bitacora": 1.0,
                "fecha_coincide": 1.0,
                "dias_diferencia": 0.0,
                "similitud_semantica": 0.9,
                "coincidencia_zona": 1.0,
                "coincidencia_actividad": 1.0,
                "similitud_descripcion": 0.8,
                "datos_fecha_disponibles": 1.0,
                "datos_incidencia_disponibles": 1.0,
                "tiene_materiales": 1.0,
            },
            feature_schema_version="1.0",
            feedback_ids=(f"fb_{i}",),
        ))
    # 5 negativos
    for i in range(5, 10):
        pairs.append(LabeledPair(
            sot_id=f"SOT_{i}",
            log_id=f"LOG_{i}",
            label=0,
            features={
                "incidencia_exacta": 0.0,
                "sot_en_bitacora": 0.0,
                "fecha_coincide": 0.0,
                "dias_diferencia": 10.0,
                "similitud_semantica": 0.1,
                "coincidencia_zona": 0.0,
                "coincidencia_actividad": 0.0,
                "similitud_descripcion": 0.1,
                "datos_fecha_disponibles": 1.0,
                "datos_incidencia_disponibles": 0.0,
                "tiene_materiales": 0.0,
            },
            feature_schema_version="1.0",
            feedback_ids=(f"fb_{i}",),
        ))
    return pairs


def test_fit_requires_both_classes(classifier):
    # Solo positivos
    pos_pairs = [
        LabeledPair(sot_id="S1", log_id="L1", label=1, features={"f1": 1.0}, feature_schema_version="1.0", feedback_ids=("fb1",))
    ]
    with pytest.raises(ValueError, match="ambas clases"):
        classifier.fit(pos_pairs, pos_pairs)


def test_fit_and_predict(classifier, sample_labeled_pairs):
    # Split manual para test
    train = sample_labeled_pairs[:8]
    val = sample_labeled_pairs[8:]

    report = classifier.fit(train, val)

    assert report.model_version.startswith("catboost_")
    assert report.train_size == 8
    assert report.val_size == 2
    assert "precision" in report.metrics
    assert "recall" in report.metrics
    assert "f1" in report.metrics
    assert "roc_auc" in report.metrics

    # Test predict_proba
    test_features = [p.features for p in sample_labeled_pairs]
    probs = classifier.predict_proba(test_features)
    assert len(probs) == 10
    assert all(0 <= p <= 1 for p in probs)
    # Positivos deberían tener mayor probabilidad
    assert np.mean(probs[:5]) > np.mean(probs[5:])


def test_save_load(classifier, sample_labeled_pairs, tmp_path):
    train = sample_labeled_pairs[:8]
    val = sample_labeled_pairs[8:]
    classifier.fit(train, val)

    model_path = tmp_path / "test_model.cbm"
    meta = classifier.save(str(model_path))

    assert model_path.exists()
    assert meta.model_version == classifier._model_version
    assert meta.status == "CANDIDATE"

    # Cargar y verificar
    loaded = CatBoostMatchingClassifier.load(str(model_path))
    probs_orig = classifier.predict_proba([p.features for p in sample_labeled_pairs])
    probs_loaded = loaded.predict_proba([p.features for p in sample_labeled_pairs])
    np.testing.assert_array_almost_equal(probs_orig, probs_loaded)


def test_metadata(classifier):
    meta = classifier.metadata()
    assert meta.model_version == "unknown"
    assert meta.status == "CANDIDATE"