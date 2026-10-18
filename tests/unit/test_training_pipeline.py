"""Tests de integración para TrainingPipeline y shadow evaluation."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sot_matcher.infrastructure import SqliteFeedbackRepository, SotLogFeatureExtractor, SotCandidateRetriever
from sot_matcher.infrastructure.embedding import get_embedder
from sot_matcher.application import TrainingPipeline, TrainConfig, FeedbackFilters
from sot_matcher.infrastructure.calibration import CalibrationConfig
from sot_matcher.infrastructure.classifier import CatBoostMatchingClassifier
from sot_matcher.infrastructure.model_registry import FileModelRegistry
from sot_matcher.domain.feedback import FeedbackDecision, FeedbackEvent, LabeledPair


@pytest.fixture
def temp_db_path():
    import uuid
    tmp_dir = Path(tempfile.gettempdir())
    db_path = tmp_dir / f"test_train_{uuid.uuid4().hex}.db"
    yield str(db_path)
    try:
        db_path.unlink(missing_ok=True)
    except PermissionError:
        pass


@pytest.fixture
def temp_models_dir(temp_db_path):
    models_dir = Path(temp_db_path).parent / "test_models"
    models_dir.mkdir(exist_ok=True)
    yield str(models_dir)
    import shutil
    try:
        shutil.rmtree(models_dir, ignore_errors=True)
    except Exception:
        pass


@pytest.fixture
def pipeline_components(temp_db_path, temp_models_dir):
    repo = SqliteFeedbackRepository(temp_db_path)
    embedder = get_embedder()
    extractor = SotLogFeatureExtractor(embedder=embedder)
    retriever = SotCandidateRetriever(embedder=embedder)
    classifier = CatBoostMatchingClassifier(iterations=20, verbose=False, random_seed=42)
    registry = FileModelRegistry(temp_db_path, temp_models_dir)
    return repo, extractor, retriever, classifier, registry


@pytest.fixture
def sample_prepared_data():
    """Datos preparados sintéticos para testing."""
    from datetime import date
    sots = pd.DataFrame([
        {"sot": "SOT_1", "source_text": "INC000123 mantenimiento zona norte 30/01/2026", "date_hint": date(2026, 1, 30)},
        {"sot": "SOT_2", "source_text": "instalacion cable fibra zona sur 15/02/2026", "date_hint": date(2026, 2, 15)},
        {"sot": "SOT_3", "source_text": "revision preventiva zona este", "date_hint": None},
        {"sot": "SOT_4", "source_text": "reparacion urgente INC000456", "date_hint": date(2026, 3, 10)},
        {"sot": "SOT_5", "source_text": "mantenimiento correctivo zona oeste", "date_hint": date(2026, 4, 5)},
        {"sot": "SOT_6", "source_text": "instalacion zona norte", "date_hint": date(2026, 5, 1)},
        {"sot": "SOT_7", "source_text": "revision zona sur", "date_hint": date(2026, 5, 15)},
        {"sot": "SOT_8", "source_text": "mantenimiento INC000789", "date_hint": date(2026, 6, 1)},
    ])
    logs = pd.DataFrame([
        {"record_id": "LOG_1", "date": date(2026, 1, 30), "log_text": "Atención INC000123 mantenimiento zona norte", "observation": "", "materials_data": "cable FO"},
        {"record_id": "LOG_2", "date": date(2026, 2, 15), "log_text": "Instalación cable fibra óptica zona sur", "observation": "", "materials_data": "conector fibra"},
        {"record_id": "LOG_3", "date": date(2026, 1, 30), "log_text": "Inspección zona este sin hallazgos", "observation": "", "materials_data": ""},
        {"record_id": "LOG_4", "date": date(2026, 3, 10), "log_text": "Reparación urgente INC000456 completada", "observation": "Materiales usados", "materials_data": "ferreteria adaptador"},
        {"record_id": "LOG_5", "date": date(2026, 4, 5), "log_text": "Mantenimiento correctivo zona oeste", "observation": "", "materials_data": ""},
        {"record_id": "LOG_6", "date": date(2026, 5, 1), "log_text": "Instalación zona norte", "observation": "", "materials_data": ""},
        {"record_id": "LOG_7", "date": date(2026, 5, 15), "log_text": "Revisión zona sur", "observation": "", "materials_data": ""},
        {"record_id": "LOG_8", "date": date(2026, 6, 1), "log_text": "Mantenimiento INC000789", "observation": "", "materials_data": ""},
    ])
    return sots, logs


def insert_feedback(repo, pairs: list[tuple[str, str, FeedbackDecision]]):
    """Helper para insertar feedback de prueba."""
    for sot_id, log_id, decision in pairs:
        repo.save_feedback(FeedbackEvent(
            sot_id=sot_id, log_id=log_id,
            decision=decision, feature_schema_version="1.0"
        ))


def test_build_dataset_with_real_records(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    insert_feedback(repo, [
        ("SOT_1", "LOG_1", FeedbackDecision.CORRECT),
        ("SOT_2", "LOG_2", FeedbackDecision.CORRECT),
        ("SOT_3", "LOG_3", FeedbackDecision.INCORRECT),
        ("SOT_4", "LOG_4", FeedbackDecision.CORRECT),
        ("SOT_5", "LOG_5", FeedbackDecision.INCORRECT),
    ])

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)
    report = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version="1.0")

    assert report.total_events == 5
    assert report.positive_pairs == 3
    assert report.negative_pairs == 2
    assert report.excluded_inconclusive == 0
    assert report.feature_schema_version == "1.0"
    assert len(report.dataset_fingerprint) == 16


def test_build_dataset_excludes_inconclusive(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    insert_feedback(repo, [
        ("SOT_1", "LOG_1", FeedbackDecision.CORRECT),
        ("SOT_2", "LOG_2", FeedbackDecision.INCONCLUSIVE),
    ])

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)
    report = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version="1.0")

    assert report.total_events == 2
    assert report.excluded_inconclusive == 1
    assert report.positive_pairs == 1
    assert report.negative_pairs == 0


def test_train_and_evaluate(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    # Insertar feedback suficiente para entrenar (mín 4 por clase para split)
    insert_feedback(repo, [
        ("SOT_1", "LOG_1", FeedbackDecision.CORRECT),
        ("SOT_2", "LOG_2", FeedbackDecision.CORRECT),
        ("SOT_3", "LOG_3", FeedbackDecision.CORRECT),
        ("SOT_4", "LOG_4", FeedbackDecision.CORRECT),
        ("SOT_5", "LOG_5", FeedbackDecision.INCORRECT),
        ("SOT_6", "LOG_6", FeedbackDecision.INCORRECT),
        ("SOT_7", "LOG_7", FeedbackDecision.INCORRECT),
        ("SOT_8", "LOG_8", FeedbackDecision.INCORRECT),
    ])

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    # Usar build_dataset para obtener pares CON features
    report = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version="1.0")

    # Obtener pares etiquetados CON features (a través de build_dataset internamente)
    # Nota: build_dataset devuelve reporte, pero no los pares. En producción se usarían los pares persistidos.
    # Para test, reconstruimos los pares con features usando la misma lógica
    events = repo.list_feedback(FeedbackFilters(limit=10000))
    labeled_events = [e for e in events if e.is_valid_for_training()]

    superseded_ids: set[str] = set()
    events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}
    for e in sorted(labeled_events, key=lambda x: x.created_at):
        key = (e.sot_id, e.log_id)
        events_by_key.setdefault(key, []).append(e)
        if e.supersedes_feedback_id:
            superseded_ids.add(e.supersedes_feedback_id)

    latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
    for key, ev_list in events_by_key.items():
        for event in reversed(ev_list):
            if event.feedback_id not in superseded_ids:
                latest_by_pair[key] = event
                break

    sots_by_id = prepared_sots.set_index("sot").to_dict("index")
    logs_by_id = prepared_logs.set_index("record_id").to_dict("index")

    pairs = []
    for (sot_id, log_id), event in latest_by_pair.items():
        sot_rec = sots_by_id.get(sot_id)
        log_rec = logs_by_id.get(log_id)
        if sot_rec is None or log_rec is None:
            continue
        features = extractor.extract(sot_rec, log_rec)
        pairs.append(LabeledPair(
            sot_id=sot_id,
            log_id=log_id,
            label=event.label(),
            features=features,
            feature_schema_version=extractor.schema_version,
            feedback_ids=(event.feedback_id,),
        ))

    # Verificar que las features se extrajeron
    assert len(pairs) == 8
    for pair in pairs:
        assert pair.features
        assert "incidencia_exacta" in pair.features

    # Entrenar
    config = TrainConfig(test_size=0.25, val_size=0.0, random_state=42, calibration=CalibrationConfig(method="none"))
    report = pipeline.train(pairs, config)

    assert report.model_version.startswith("catboost_")
    assert report.train_size > 0
    assert report.test_size > 0
    assert "precision" in report.metrics

    # Evaluar usando test set persistido (usar fingerprint del reporte)
    test_fingerprint = report.config.get("test_fingerprint")
    test_pairs = pipeline.load_test_set(test_fingerprint)
    eval_report = pipeline.evaluate(classifier, test_pairs)

    assert "precision" in eval_report.metrics
    assert "recall" in eval_report.metrics
    assert "ece" in eval_report.calibration_metrics
    assert eval_report.recommendation in ("ACTIVATE", "REJECT", "NEEDS_MORE_DATA")


def test_shadow_evaluation(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    # Insertar feedback (4 positivos, 4 negativos = 8 total)
    insert_feedback(repo, [
        ("SOT_1", "LOG_1", FeedbackDecision.CORRECT),
        ("SOT_2", "LOG_2", FeedbackDecision.CORRECT),
        ("SOT_3", "LOG_3", FeedbackDecision.CORRECT),
        ("SOT_4", "LOG_4", FeedbackDecision.CORRECT),
        ("SOT_5", "LOG_5", FeedbackDecision.INCORRECT),
        ("SOT_6", "LOG_6", FeedbackDecision.INCORRECT),
        ("SOT_7", "LOG_7", FeedbackDecision.INCORRECT),
        ("SOT_8", "LOG_8", FeedbackDecision.INCORRECT),
    ])

    # Entrenar modelo rápido
    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    # Reconstruir pares con features para entrenar
    events = repo.list_feedback(FeedbackFilters(limit=10000))
    labeled_events = [e for e in events if e.is_valid_for_training()]
    superseded_ids: set[str] = set()
    events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}
    for e in sorted(labeled_events, key=lambda x: x.created_at):
        key = (e.sot_id, e.log_id)
        events_by_key.setdefault(key, []).append(e)
        if e.supersedes_feedback_id:
            superseded_ids.add(e.supersedes_feedback_id)

    latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
    for key, ev_list in events_by_key.items():
        for event in reversed(ev_list):
            if event.feedback_id not in superseded_ids:
                latest_by_pair[key] = event
                break

    sots_by_id = prepared_sots.set_index("sot").to_dict("index")
    logs_by_id = prepared_logs.set_index("record_id").to_dict("index")

    pairs = []
    for (sot_id, log_id), event in latest_by_pair.items():
        sot_rec = sots_by_id.get(sot_id)
        log_rec = logs_by_id.get(log_id)
        if sot_rec is None or log_rec is None:
            continue
        features = extractor.extract(sot_rec, log_rec)
        pairs.append(LabeledPair(
            sot_id=sot_id,
            log_id=log_id,
            label=event.label(),
            features=features,
            feature_schema_version=extractor.schema_version,
            feedback_ids=(event.feedback_id,),
        ))

    config = TrainConfig(test_size=0.25, val_size=0.0, random_state=42, calibration=CalibrationConfig(method="none"))
    pipeline.train(pairs, config)

    # Mock results_df del motor actual
    results_df = pd.DataFrame([
        {"sot": "SOT_1", "estado": "Exacto", "confianza": 1.0},
        {"sot": "SOT_2", "estado": "Exacto", "confianza": 1.0},
        {"sot": "SOT_3", "estado": "Pendiente", "confianza": None},
        {"sot": "SOT_4", "estado": "Semántico", "confianza": 0.82},
        {"sot": "SOT_5", "estado": "Pendiente", "confianza": None},
        {"sot": "SOT_6", "estado": "Pendiente", "confianza": None},
        {"sot": "SOT_7", "estado": "Pendiente", "confianza": None},
        {"sot": "SOT_8", "estado": "Pendiente", "confianza": None},
    ])

    # Shadow evaluation
    shadow_results = pipeline.run_shadow_evaluation(prepared_sots, prepared_logs, results_df, classifier)

    assert len(shadow_results) == 8  # 8 pares con feedback
    for r in shadow_results:
        assert r.true_label in (0, 1)
        assert r.engine_prediction in ("Exacto", "Semántico", "Pendiente")
        assert 0 <= r.classifier_probability <= 1
        assert r.classifier_prediction in (0, 1)


def test_dataset_fingerprint_deterministic(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    insert_feedback(repo, [("SOT_1", "LOG_1", FeedbackDecision.CORRECT)])

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    report1 = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version="1.0")
    report2 = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version="1.0")

    assert report1.dataset_fingerprint == report2.dataset_fingerprint


def test_persist_and_load_test_set(pipeline_components, sample_prepared_data):
    repo, extractor, retriever, classifier, registry = pipeline_components
    prepared_sots, prepared_logs = sample_prepared_data

    # Insertar feedback (8 pares = 4 pos, 4 neg)
    insert_feedback(repo, [
        ("SOT_1", "LOG_1", FeedbackDecision.CORRECT),
        ("SOT_2", "LOG_2", FeedbackDecision.CORRECT),
        ("SOT_3", "LOG_3", FeedbackDecision.CORRECT),
        ("SOT_4", "LOG_4", FeedbackDecision.CORRECT),
        ("SOT_5", "LOG_5", FeedbackDecision.INCORRECT),
        ("SOT_6", "LOG_6", FeedbackDecision.INCORRECT),
        ("SOT_7", "LOG_7", FeedbackDecision.INCORRECT),
        ("SOT_8", "LOG_8", FeedbackDecision.INCORRECT),
    ])

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    # Reconstruir pares con features
    events = repo.list_feedback(FeedbackFilters(limit=10000))
    labeled_events = [e for e in events if e.is_valid_for_training()]
    superseded_ids: set[str] = set()
    events_by_key: dict[tuple[str, str], list[FeedbackEvent]] = {}
    for e in sorted(labeled_events, key=lambda x: x.created_at):
        key = (e.sot_id, e.log_id)
        events_by_key.setdefault(key, []).append(e)
        if e.supersedes_feedback_id:
            superseded_ids.add(e.supersedes_feedback_id)

    latest_by_pair: dict[tuple[str, str], FeedbackEvent] = {}
    for key, ev_list in events_by_key.items():
        for event in reversed(ev_list):
            if event.feedback_id not in superseded_ids:
                latest_by_pair[key] = event
                break

    sots_by_id = prepared_sots.set_index("sot").to_dict("index")
    logs_by_id = prepared_logs.set_index("record_id").to_dict("index")

    pairs = []
    for (sot_id, log_id), event in latest_by_pair.items():
        sot_rec = sots_by_id.get(sot_id)
        log_rec = logs_by_id.get(log_id)
        if sot_rec is None or log_rec is None:
            continue
        features = extractor.extract(sot_rec, log_rec)
        pairs.append(LabeledPair(
            sot_id=sot_id,
            log_id=log_id,
            label=event.label(),
            features=features,
            feature_schema_version=extractor.schema_version,
            feedback_ids=(event.feedback_id,),
        ))

    # Entrenar (esto persiste el test set)
    config = TrainConfig(test_size=0.25, val_size=0.0, random_state=42, calibration=CalibrationConfig(method="none"))
    report = pipeline.train(pairs, config)

    # Cargar test set usando fingerprint del reporte
    test_fingerprint = report.config.get("test_fingerprint")
    test_pairs = pipeline.load_test_set(test_fingerprint)

    assert len(test_pairs) == report.test_size
    for p in test_pairs:
        assert p.features
        assert p.label in (0, 1)