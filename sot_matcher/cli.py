"""CLI para entrenamiento y evaluación de modelos supervisados."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from sot_matcher.infrastructure import (
    SqliteFeedbackRepository,
    SotLogFeatureExtractor,
    SotCandidateRetriever,
    FileModelRegistry,
    CatBoostMatchingClassifier,
)
from sot_matcher.application import (
    TrainingPipeline,
    TrainConfig,
    FeedbackService,
    FeedbackFilters,
)
from sot_matcher.infrastructure.embedding import get_embedder
from sot_matcher.infrastructure.excel_reader import prepare_sots, prepare_logs, read_excel


def build_pipeline_components(db_path: str = "sot_matcher_feedback.db", models_dir: str = "models"):
    """Construye componentes compartidos."""
    repo = SqliteFeedbackRepository(db_path)
    embedder = get_embedder()
    extractor = SotLogFeatureExtractor(embedder=embedder)
    retriever = SotCandidateRetriever(embedder=embedder)
    classifier = CatBoostMatchingClassifier()
    registry = FileModelRegistry(db_path, models_dir)
    return repo, extractor, retriever, classifier, registry


def load_prepared_data(sot_file: str, log_file: str, sot_sheet: str = None, log_sheet: str = None):
    """Carga y prepara datos SOT y bitácoras desde archivos Excel."""
    sot_df = read_excel(sot_file, sot_sheet) if sot_sheet else read_excel(sot_file, 0)
    log_df = read_excel(log_file, log_sheet) if log_sheet else read_excel(log_file, 0)
    return prepare_sots(sot_df), prepare_logs(log_df)


def cmd_train(args: argparse.Namespace) -> int:
    """Entrena un nuevo modelo."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    # Cargar datos preparados si se proporcionan archivos
    if args.sot_file and args.log_file:
        print(f"📂 Cargando datos desde {args.sot_file} y {args.log_file}...")
        prepared_sots, prepared_logs = load_prepared_data(
            args.sot_file, args.log_file, args.sot_sheet, args.log_sheet
        )
        print(f"   SOTs: {len(prepared_sots)}, Bitácoras: {len(prepared_logs)}")

        # Construir dataset con datos reales
        print(f"📊 Construyendo dataset v{args.version}...")
        dataset_report = pipeline.build_dataset(
            prepared_sots, prepared_logs, dataset_version=args.version
        )
    else:
        # Fallback: solo feedback sin features (para compatibilidad)
        print(f"📊 Construyendo dataset v{args.version} (solo feedback)...")
        dataset_report = pipeline.build_dataset(dataset_version=args.version)

    print(f"   Eventos totales: {dataset_report.total_events}")
    print(f"   Pares positivos: {dataset_report.positive_pairs}")
    print(f"   Pares negativos: {dataset_report.negative_pairs}")
    print(f"   Excluidos (inconclusos): {dataset_report.excluded_inconclusive}")
    print(f"   Conflictos: {dataset_report.excluded_conflicts}")
    print(f"   Balance de clases: {dataset_report.class_balance:.2%}")
    print(f"   Fingerprint: {dataset_report.dataset_fingerprint}")

    if dataset_report.positive_pairs < args.min_positives:
        print(f"❌ Insuficientes positivos ({dataset_report.positive_pairs} < {args.min_positives})")
        return 1
    if dataset_report.negative_pairs < args.min_negatives:
        print(f"❌ Insuficientes negativos ({dataset_report.negative_pairs} < {args.min_negatives})")
        return 1

    # Obtener pares etiquetados CON features
    labeled_pairs = repo.get_labeled_pairs(args.version)
    if not labeled_pairs:
        print("❌ No hay pares etiquetados válidos")
        return 1

    print(f"🔧 Entrenando modelo con {len(labeled_pairs)} pares...")
    config = TrainConfig(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
        calibrate=not args.no_calibrate,
        calibration_method=args.calibration_method,
    )
    report = pipeline.train(labeled_pairs, config)

    print(f"✅ Entrenamiento completado: {report.model_version}")
    print(f"   Train: {report.train_size}, Val: {report.val_size}, Test: {report.test_size}")
    print(f"   Precision: {report.metrics.get('precision', 0):.4f}")
    print(f"   Recall: {report.metrics.get('recall', 0):.4f}")
    print(f"   F1: {report.metrics.get('f1', 0):.4f}")
    print(f"   ROC-AUC: {report.metrics.get('roc_auc', 0):.4f}")

    # Registrar como candidato
    meta = pipeline.register_candidate(classifier, report)
    print(f"📦 Modelo registrado: {meta.model_version} (status: {meta.status})")

    if args.activate:
        if report.metrics.get("precision", 0) >= args.precision_threshold and \
           report.metrics.get("recall", 0) >= args.recall_threshold:
            registry.activate(meta.model_version, "APPROVED")
            print(f"🚀 Modelo ACTIVADO en producción")
        else:
            print(f"⚠️ No cumple umbrales de activación (precision>={args.precision_threshold}, recall>={args.recall_threshold})")

    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    """Evalúa un modelo contra el motor actual."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    # Cargar modelo
    model_path = Path(args.models_dir) / f"{args.model}.cbm"
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return 1

    model = CatBoostMatchingClassifier.load(str(model_path))
    print(f"📥 Modelo cargado: {model.metadata().model_version}")

    # Cargar test set persistido si existe, sino usar todos los pares
    try:
        # Intentar cargar test set por fingerprint del modelo
        test_pairs = pipeline.load_test_set(model.metadata().training_dataset_fingerprint)
        print(f"📊 Evaluando con test set persistido ({len(test_pairs)} pares)...")
    except (FileNotFoundError, AttributeError):
        # Fallback: usar todos los pares etiquetados
        labeled_pairs = repo.get_labeled_pairs(args.version)
        if not labeled_pairs:
            print("❌ No hay pares etiquetados para evaluar")
            return 1
        test_pairs = labeled_pairs
        print(f"📊 Evaluando con {len(test_pairs)} pares (fallback)...")

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)
    eval_report = pipeline.evaluate(model, test_pairs)

    print(f"📈 Resultados evaluación:")
    print(f"   Precision: {eval_report.metrics.get('precision', 0):.4f}")
    print(f"   Recall: {eval_report.metrics.get('recall', 0):.4f}")
    print(f"   F1: {eval_report.metrics.get('f1', 0):.4f}")
    print(f"   ROC-AUC: {eval_report.metrics.get('roc_auc', 0):.4f}")
    print(f"   PR-AUC: {eval_report.metrics.get('pr_auc', 0):.4f}")
    print(f"   Brier Score: {eval_report.metrics.get('brier_score', 0):.4f}")
    print(f"   Log Loss: {eval_report.metrics.get('log_loss', 0):.4f}")
    if eval_report.calibration_metrics:
        print(f"   ECE: {eval_report.calibration_metrics.get('ece', 0):.4f}")
        print(f"   MCE: {eval_report.calibration_metrics.get('mce', 0):.4f}")
    if eval_report.baseline_metrics:
        print(f"   🔄 Comparación vs motor actual:")
        for k, v in eval_report.baseline_metrics.items():
            print(f"      {k}: {v:.4f}")
    print(f"   Recomendación: {eval_report.recommendation}")

    if eval_report.recommendation == "ACTIVATE" and args.activate:
        registry.activate(model.metadata().model_version, "APPROVED")
        print(f"🚀 Modelo ACTIVADO")

    return 0


def cmd_shadow_eval(args: argparse.Namespace) -> int:
    """Ejecuta evaluación en sombra: compara clasificador vs motor actual."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    # Cargar modelo
    model_path = Path(args.models_dir) / f"{args.model}.cbm"
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        return 1

    model = CatBoostMatchingClassifier.load(str(model_path))
    print(f"📥 Modelo cargado: {model.metadata().model_version}")

    # Cargar datos preparados
    if not args.sot_file or not args.log_file:
        print("❌ Se requieren --sot-file y --log-file para evaluación en sombra")
        return 1

    print(f"📂 Cargando datos para shadow evaluation...")
    prepared_sots, prepared_logs = load_prepared_data(
        args.sot_file, args.log_file, args.sot_sheet, args.log_sheet
    )

    # Cargar resultados del motor actual
    from sot_matcher import CrossingPipeline, MatchingSettings
    pipeline = CrossingPipeline()
    prepared = pipeline.prepare(
        Path(args.sot_file).read_bytes(),
        Path(args.log_file).read_bytes(),
        use_column_dates=True,
        sot_sheet=args.sot_sheet,
        log_sheet=args.log_sheet,
    )
    results = pipeline.cross(prepared, settings=MatchingSettings(threshold=0.78))
    results_df = results.results

    # Ejecutar shadow evaluation
    training_pipeline = TrainingPipeline(repo, extractor, classifier, registry)
    print(f"🌑 Ejecutando shadow evaluation...")
    shadow_results = training_pipeline.run_shadow_evaluation(
        prepared_sots, prepared_logs, results_df, model
    )

    if not shadow_results:
        print("❌ No hay casos con feedback para comparar")
        return 1

    print(f"📊 Resultados shadow evaluation ({len(shadow_results)} casos con feedback):")

    # Métricas comparativas
    y_true = [r.true_label for r in shadow_results]
    y_pred_engine = [1 if r.engine_prediction in ("Exacto", "Semántico") else 0 for r in shadow_results]
    y_pred_clf = [r.classifier_prediction for r in shadow_results]

    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

    engine_precision = precision_score(y_true, y_pred_engine, zero_division=0)
    engine_recall = recall_score(y_true, y_pred_engine, zero_division=0)
    engine_f1 = f1_score(y_true, y_pred_engine, zero_division=0)
    engine_accuracy = accuracy_score(y_true, y_pred_engine)

    clf_precision = precision_score(y_true, y_pred_clf, zero_division=0)
    clf_recall = recall_score(y_true, y_pred_clf, zero_division=0)
    clf_f1 = f1_score(y_true, y_pred_clf, zero_division=0)
    clf_accuracy = accuracy_score(y_true, y_pred_clf)

    print(f"\n📈 Motor actual (CrossingPipeline):")
    print(f"   Precision: {engine_precision:.4f}")
    print(f"   Recall: {engine_recall:.4f}")
    print(f"   F1: {engine_f1:.4f}")
    print(f"   Accuracy: {engine_accuracy:.4f}")

    print(f"\n🤖 Clasificador supervisado:")
    print(f"   Precision: {clf_precision:.4f}")
    print(f"   Recall: {clf_recall:.4f}")
    print(f"   F1: {clf_f1:.4f}")
    print(f"   Accuracy: {clf_accuracy:.4f}")

    # Casos donde discrepan
    disagreements = sum(1 for r in shadow_results if r.engine_prediction != "Pendiente" and
                       ((r.engine_prediction in ("Exacto", "Semántico")) != bool(r.classifier_prediction)))
    print(f"\n⚠️  Discrepancias: {disagreements}/{len(shadow_results)} casos")

    # Guardar reporte detallado si se pide
    if args.output:
        import csv
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["sot_id", "log_id", "true_label", "engine_pred", "engine_conf",
                           "clf_proba", "clf_pred", "agrees"])
            for r in shadow_results:
                agrees = (r.engine_prediction in ("Exacto", "Semántico")) == bool(r.classifier_prediction)
                writer.writerow([r.sot_id, r.log_id, r.true_label, r.engine_prediction,
                               r.engine_confidence, r.classifier_probability, r.classifier_prediction, agrees])
        print(f"💾 Reporte guardado en: {args.output}")

    return 0


def cmd_list_models(args: argparse.Namespace) -> int:
    """Lista modelos registrados."""
    repo, _, _, _, registry = build_pipeline_components(args.db, args.models_dir)
    models = registry.list_models(args.status)

    if not models:
        print("No hay modelos registrados.")
        return 0

    print(f"{'Versión':<25} {'Estado':<12} {'Precision':<10} {'Recall':<10} {'F1':<8} {'Creado'}")
    print("-" * 85)
    for m in models:
        print(f"{m.model_version:<25} {m.status:<12} {m.metrics.get('precision', 0):<10.4f} {m.metrics.get('recall', 0):<10.4f} {m.metrics.get('f1', 0):<8.4f} {m.created_at[:16]}")

    return 0


def cmd_activate(args: argparse.Namespace) -> int:
    """Activa un modelo en producción."""
    repo, _, _, _, registry = build_pipeline_components(args.db, args.models_dir)
    registry.activate(args.model, "APPROVED")
    print(f"✅ Modelo {args.model} activado")
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Revierte al modelo anterior."""
    repo, _, _, _, registry = build_pipeline_components(args.db, args.models_dir)
    registry.rollback(args.model)
    print(f"🔄 Rollback a {args.model} completado")
    return 0


def cmd_build_dataset(args: argparse.Namespace) -> int:
    """Construye y muestra reporte de dataset sin entrenar."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    if args.sot_file and args.log_file:
        prepared_sots, prepared_logs = load_prepared_data(
            args.sot_file, args.log_file, args.sot_sheet, args.log_sheet
        )
        report = pipeline.build_dataset(prepared_sots, prepared_logs, dataset_version=args.version)
    else:
        report = pipeline.build_dataset(dataset_version=args.version)

    print(f"📊 Dataset Report v{report.version}")
    print(f"   Fingerprint: {report.dataset_fingerprint}")
    print(f"   Eventos totales: {report.total_events}")
    print(f"   Pares positivos: {report.positive_pairs}")
    print(f"   Pares negativos: {report.negative_pairs}")
    print(f"   Excluidos (inconclusos): {report.excluded_inconclusive}")
    print(f"   Conflictos: {report.excluded_conflicts}")
    print(f"   Schema features: {report.feature_schema_version}")
    print(f"   Balance: {report.class_balance:.2%}")

    return 0


def cmd_active_export(args: argparse.Namespace) -> int:
    """Exporta casos PENDIENTE para anotación humana (Active Learning)."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    # Cargar datos preparados
    print(f"📂 Cargando datos desde {args.sot_file} y {args.log_file}...")
    prepared_sots, prepared_logs = load_prepared_data(
        args.sot_file, args.log_file, args.sot_sheet, args.log_sheet
    )
    print(f"   SOTs: {len(prepared_sots)}, Bitácoras: {len(prepared_logs)}")

    # Ejecutar cruce para obtener resultados del motor actual
    from sot_matcher import CrossingPipeline, MatchingSettings
    pipeline = CrossingPipeline()
    prepared = pipeline.prepare(
        Path(args.sot_file).read_bytes(),
        Path(args.log_file).read_bytes(),
        use_column_dates=True,
        sot_sheet=args.sot_sheet,
        log_sheet=args.log_sheet,
    )
    results = pipeline.cross(prepared, settings=MatchingSettings(threshold=0.78))
    results_df = results.results

    # Configurar active learning
    from sot_matcher.infrastructure.active_learning import (
        ActiveLearningLoop,
        ActiveLearningConfig,
        ActiveLearningStrategy,
    )

    al_loop = ActiveLearningLoop(repo, extractor, retriever, classifier, registry, None)

    strategy_map = {
        "uncertainty": ActiveLearningStrategy.UNCERTAINTY,
        "margin": ActiveLearningStrategy.MARGIN,
        "entropy": ActiveLearningStrategy.ENTROPY,
        "diversity": ActiveLearningStrategy.DIVERSITY,
        "hybrid": ActiveLearningStrategy.HYBRID,
        "least_confident": ActiveLearningStrategy.LEAST_CONFIDENT,
        "disagreement": ActiveLearningStrategy.DISAGREEMENT,
    }

    config = ActiveLearningConfig(
        strategy=strategy_map[args.strategy],
        batch_size=args.batch_size,
        top_k_candidates=args.top_k,
    )

    print(f"🔍 Exportando casos PENDIENTE con estrategia '{args.strategy}'...")
    report = al_loop.export_pending_for_annotation(
        prepared_sots, prepared_logs, results_df, config, args.output
    )

    print(f"✅ Exportación completada:")
    print(f"   Iteración: {report.iteration}")
    print(f"   Casos exportados: {report.exported_count}")
    print(f"   Estrategia: {report.strategy.value}")
    print(f"   Incertidumbre media: {report.uncertainty_stats.get('mean', 0):.3f}")
    print(f"   Archivos generados: {args.output}.csv, {args.output}.json")

    return 0


def cmd_active_import(args: argparse.Namespace) -> int:
    """Importa anotaciones desde CSV/JSON."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    from sot_matcher.infrastructure.active_learning import ActiveLearningLoop
    al_loop = ActiveLearningLoop(repo, extractor, retriever, classifier, registry, None)

    print(f"📥 Importando anotaciones desde {args.file}...")
    count = al_loop.import_annotations(args.file, args.reviewer)
    print(f"✅ {count} anotaciones importadas")

    return 0


def cmd_active_retrain(args: argparse.Namespace) -> int:
    """Re-entrena modelo con nuevos datos de feedback."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    from sot_matcher.infrastructure.active_learning import ActiveLearningLoop
    al_loop = ActiveLearningLoop(repo, extractor, retriever, classifier, registry, None)

    print("🔄 Re-entrenando modelo con feedback actualizado...")
    try:
        pipeline = al_loop.retrain_incremental()
        print(f"✅ Re-entrenamiento completado")

        if args.activate:
            # El modelo se registra como CANDIDATE, activar manualmente si se desea
            print("ℹ️ Modelo registrado como CANDIDATE. Usa 'activate' para poner en producción.")

        return 0
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1
    parser = argparse.ArgumentParser(
        prog="sot-matcher-feedback",
        description="CLI para feedback supervisado en SOT Matcher",
    )
    parser.add_argument("--db", default="sot_matcher_feedback.db", help="Ruta a SQLite DB")
    parser.add_argument("--models-dir", default="models", help="Directorio de modelos")
    parser.add_argument("--version", default="1.0", help="Versión del esquema de features")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_p = subparsers.add_parser("train", help="Entrenar nuevo modelo")
    train_p.add_argument("--sot-file", help="Archivo Excel SOT (para features reales)")
    train_p.add_argument("--log-file", help="Archivo Excel bitácoras (para features reales)")
    train_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    train_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    train_p.add_argument("--min-positives", type=int, default=10, help="Mínimo pares positivos")
    train_p.add_argument("--min-negatives", type=int, default=10, help="Mínimo pares negativos")
    train_p.add_argument("--test-size", type=float, default=0.2, help="Tamaño test set")
    train_p.add_argument("--val-size", type=float, default=0.1, help="Tamaño validation set")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--no-calibrate", action="store_true", help="Desactivar calibración de probabilidades")
    train_p.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="isotonic")
    train_p.add_argument("--precision-threshold", type=float, default=0.85, help="Umbral precision para activar")
    train_p.add_argument("--recall-threshold", type=float, default=0.70, help="Umbral recall para activar")
    train_p.add_argument("--activate", action="store_true", help="Activar si cumple umbrales")
    train_p.set_defaults(func=cmd_train)

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluar modelo existente")
    eval_p.add_argument("model", help="Versión del modelo a evaluar")
    eval_p.add_argument("--activate", action="store_true", help="Activar si recomendación es ACTIVATE")
    eval_p.set_defaults(func=cmd_evaluate)

    # shadow-eval
    shadow_p = subparsers.add_parser("shadow-eval", help="Evaluación en sombra vs motor actual")
    shadow_p.add_argument("model", help="Versión del modelo a evaluar")
    shadow_p.add_argument("--sot-file", required=True, help="Archivo Excel SOT")
    shadow_p.add_argument("--log-file", required=True, help="Archivo Excel bitácoras")
    shadow_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    shadow_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    shadow_p.add_argument("--output", help="Archivo CSV para reporte detallado")
    shadow_p.set_defaults(func=cmd_shadow_eval)

    # build-dataset
    build_p = subparsers.add_parser("build-dataset", help="Construir reporte de dataset sin entrenar")
    build_p.add_argument("--sot-file", help="Archivo Excel SOT")
    build_p.add_argument("--log-file", help="Archivo Excel bitácoras")
    build_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    build_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    build_p.set_defaults(func=cmd_build_dataset)

    # active-learning export
    al_export_p = subparsers.add_parser("active-export", help="Exportar casos PENDIENTE para anotación (Active Learning)")
    al_export_p.add_argument("--sot-file", required=True, help="Archivo Excel SOT")
    al_export_p.add_argument("--log-file", required=True, help="Archivo Excel bitácoras")
    al_export_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    al_export_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    al_export_p.add_argument("--output", required=True, help="Ruta base de salida (se crea .csv y .json)")
    al_export_p.add_argument("--strategy", choices=["uncertainty", "margin", "entropy", "diversity", "hybrid", "least_confident", "disagreement"], default="hybrid", help="Estrategia de selección")
    al_export_p.add_argument("--batch-size", type=int, default=50, help="Casos a exportar")
    al_export_p.add_argument("--top-k", type=int, default=5, help="Candidatos por SOT")
    al_export_p.set_defaults(func=cmd_active_export)

    # active-learning import
    al_import_p = subparsers.add_parser("active-import", help="Importar anotaciones desde CSV/JSON")
    al_import_p.add_argument("file", help="Archivo CSV o JSON con anotaciones")
    al_import_p.add_argument("--reviewer", default="active_learning", help="ID del revisor")
    al_import_p.set_defaults(func=cmd_active_import)

    # active-learning retrain
    al_retrain_p = subparsers.add_parser("active-retrain", help="Re-entrenar modelo con nuevos datos")
    al_retrain_p.add_argument("--sot-file", help="Archivo Excel SOT")
    al_retrain_p.add_argument("--log-file", help="Archivo Excel bitácoras")
    al_retrain_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    al_retrain_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    al_retrain_p.add_argument("--activate", action="store_true", help="Activar si cumple umbrales")
    al_retrain_p.set_defaults(func=cmd_active_retrain)

    # list
    list_p = subparsers.add_parser("list", help="Listar modelos registrados")
    list_p.add_argument("--status", choices=["CANDIDATE", "APPROVED", "ACTIVE", "REJECTED", "RETIRED"], help="Filtrar por estado")
    list_p.set_defaults(func=cmd_list_models)

    # activate
    act_p = subparsers.add_parser("activate", help="Activar modelo en producción")
    act_p.add_argument("model", help="Versión del modelo a activar")
    act_p.set_defaults(func=cmd_activate)

    # rollback
    rb_p = subparsers.add_parser("rollback", help="Rollback a modelo anterior")
    rb_p.add_argument("model", help="Versión del modelo anterior")
    rb_p.set_defaults(func=cmd_rollback)

    args = parser.parse_args()
    return args.func(args)


def main() -> int:
    """Entry point for CLI."""
    parser = argparse.ArgumentParser(
        prog="sot-matcher-feedback",
        description="CLI para feedback supervisado en SOT Matcher",
    )
    parser.add_argument("--db", default="sot_matcher_feedback.db", help="Ruta a SQLite DB")
    parser.add_argument("--models-dir", default="models", help="Directorio de modelos")
    parser.add_argument("--version", default="1.0", help="Versión del esquema de features")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    train_p = subparsers.add_parser("train", help="Entrenar nuevo modelo")
    train_p.add_argument("--sot-file", help="Archivo Excel SOT (para features reales)")
    train_p.add_argument("--log-file", help="Archivo Excel bitácoras (para features reales)")
    train_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    train_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    train_p.add_argument("--min-positives", type=int, default=10, help="Mínimo pares positivos")
    train_p.add_argument("--min-negatives", type=int, default=10, help="Mínimo pares negativos")
    train_p.add_argument("--test-size", type=float, default=0.2, help="Tamaño test set")
    train_p.add_argument("--val-size", type=float, default=0.1, help="Tamaño validation set")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--no-calibrate", action="store_true", help="Desactivar calibración de probabilidades")
    train_p.add_argument("--calibration-method", choices=["isotonic", "platt", "none"], default="isotonic")
    train_p.add_argument("--precision-threshold", type=float, default=0.85, help="Umbral precision para activar")
    train_p.add_argument("--recall-threshold", type=float, default=0.70, help="Umbral recall para activar")
    train_p.add_argument("--activate", action="store_true", help="Activar si cumple umbrales")
    train_p.set_defaults(func=cmd_train)

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluar modelo existente")
    eval_p.add_argument("model", help="Versión del modelo a evaluar")
    eval_p.add_argument("--activate", action="store_true", help="Activar si recomendación es ACTIVATE")
    eval_p.set_defaults(func=cmd_evaluate)

    # shadow-eval
    shadow_p = subparsers.add_parser("shadow-eval", help="Evaluación en sombra vs motor actual")
    shadow_p.add_argument("model", help="Versión del modelo a evaluar")
    shadow_p.add_argument("--sot-file", required=True, help="Archivo Excel SOT")
    shadow_p.add_argument("--log-file", required=True, help="Archivo Excel bitácoras")
    shadow_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    shadow_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    shadow_p.add_argument("--output", help="Archivo CSV para reporte detallado")
    shadow_p.set_defaults(func=cmd_shadow_eval)

    # build-dataset
    build_p = subparsers.add_parser("build-dataset", help="Construir reporte de dataset sin entrenar")
    build_p.add_argument("--sot-file", help="Archivo Excel SOT")
    build_p.add_argument("--log-file", help="Archivo Excel bitácoras")
    build_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    build_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    build_p.set_defaults(func=cmd_build_dataset)

    # active-learning export
    al_export_p = subparsers.add_parser("active-export", help="Exportar casos PENDIENTE para anotación (Active Learning)")
    al_export_p.add_argument("--sot-file", required=True, help="Archivo Excel SOT")
    al_export_p.add_argument("--log-file", required=True, help="Archivo Excel bitácoras")
    al_export_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    al_export_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    al_export_p.add_argument("--output", required=True, help="Ruta base de salida (se crea .csv y .json)")
    al_export_p.add_argument("--strategy", choices=["uncertainty", "margin", "entropy", "diversity", "hybrid", "least_confident", "disagreement"], default="hybrid", help="Estrategia de selección")
    al_export_p.add_argument("--batch-size", type=int, default=50, help="Casos a exportar")
    al_export_p.add_argument("--top-k", type=int, default=5, help="Candidatos por SOT")
    al_export_p.set_defaults(func=cmd_active_export)

    # active-learning import
    al_import_p = subparsers.add_parser("active-import", help="Importar anotaciones desde CSV/JSON")
    al_import_p.add_argument("file", help="Archivo CSV o JSON con anotaciones")
    al_import_p.add_argument("--reviewer", default="active_learning", help="ID del revisor")
    al_import_p.set_defaults(func=cmd_active_import)

    # active-learning retrain
    al_retrain_p = subparsers.add_parser("active-retrain", help="Re-entrenar modelo con nuevos datos")
    al_retrain_p.add_argument("--sot-file", help="Archivo Excel SOT")
    al_retrain_p.add_argument("--log-file", help="Archivo Excel bitácoras")
    al_retrain_p.add_argument("--sot-sheet", help="Hoja específica SOT")
    al_retrain_p.add_argument("--log-sheet", help="Hoja específica bitácoras")
    al_retrain_p.add_argument("--activate", action="store_true", help="Activar si cumple umbrales")
    al_retrain_p.set_defaults(func=cmd_active_retrain)

    # list
    list_p = subparsers.add_parser("list", help="Listar modelos registrados")
    list_p.add_argument("--status", choices=["CANDIDATE", "APPROVED", "ACTIVE", "REJECTED", "RETIRED"], help="Filtrar por estado")
    list_p.set_defaults(func=cmd_list_models)

    # activate
    act_p = subparsers.add_parser("activate", help="Activar modelo en producción")
    act_p.add_argument("model", help="Versión del modelo a activar")
    act_p.set_defaults(func=cmd_activate)

    # rollback
    rb_p = subparsers.add_parser("rollback", help="Rollback a modelo anterior")
    rb_p.add_argument("model", help="Versión del modelo anterior")
    rb_p.set_defaults(func=cmd_rollback)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())