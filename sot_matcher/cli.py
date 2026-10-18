"""CLI para entrenamiento y evaluación de modelos supervisados."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def build_pipeline_components(db_path: str = "sot_matcher_feedback.db", models_dir: str = "models"):
    """Construye componentes compartidos."""
    repo = SqliteFeedbackRepository(db_path)
    embedder = get_embedder()
    extractor = SotLogFeatureExtractor(embedder=embedder)
    retriever = SotCandidateRetriever(embedder=embedder)
    classifier = CatBoostMatchingClassifier()
    registry = FileModelRegistry(db_path, models_dir)
    return repo, extractor, retriever, classifier, registry


def cmd_train(args: argparse.Namespace) -> int:
    """Entrena un nuevo modelo."""
    repo, extractor, retriever, classifier, registry = build_pipeline_components(
        args.db, args.models_dir
    )

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)

    # Construir dataset
    print(f"📊 Construyendo dataset v{args.version}...")
    dataset_report = pipeline.build_dataset(dataset_version=args.version)
    print(f"   Eventos totales: {dataset_report.total_events}")
    print(f"   Pares positivos: {dataset_report.positive_pairs}")
    print(f"   Pares negativos: {dataset_report.negative_pairs}")
    print(f"   Excluidos (inconclusos): {dataset_report.excluded_inconclusive}")
    print(f"   Balance de clases: {dataset_report.class_balance:.2%}")

    if dataset_report.positive_pairs < args.min_positives:
        print(f"❌ Insuficientes positivos ({dataset_report.positive_pairs} < {args.min_positives})")
        return 1
    if dataset_report.negative_pairs < args.min_negatives:
        print(f"❌ Insuficientes negativos ({dataset_report.negative_pairs} < {args.min_negatives})")
        return 1

    # Obtener pares etiquetados
    labeled_pairs = repo.get_labeled_pairs(args.version)
    if not labeled_pairs:
        print("❌ No hay pares etiquetados válidos")
        return 1

    print(f"🔧 Entrenando modelo con {len(labeled_pairs)} pares...")
    config = TrainConfig(
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.seed,
    )
    report = pipeline.train(labeled_pairs, config)

    print(f"✅ Entrenamiento completado: {report.model_version}")
    print(f"   Train: {report.train_size}, Val: {report.val_size}")
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

    # Obtener test set (simplificado: usar todos los pares etiquetados)
    labeled_pairs = repo.get_labeled_pairs(args.version)
    if not labeled_pairs:
        print("❌ No hay pares etiquetados para evaluar")
        return 1

    # Nota: En producción usar test set separado persistido durante entrenamiento
    print(f"📊 Evaluando con {len(labeled_pairs)} pares...")

    pipeline = TrainingPipeline(repo, extractor, classifier, registry)
    eval_report = pipeline.evaluate(model, labeled_pairs)

    print(f"📈 Resultados evaluación:")
    print(f"   Precision: {eval_report.metrics.get('precision', 0):.4f}")
    print(f"   Recall: {eval_report.metrics.get('recall', 0):.4f}")
    print(f"   F1: {eval_report.metrics.get('f1', 0):.4f}")
    print(f"   ROC-AUC: {eval_report.metrics.get('roc_auc', 0):.4f}")
    print(f"   PR-AUC: {eval_report.metrics.get('pr_auc', 0):.4f}")
    print(f"   Recomendación: {eval_report.recommendation}")

    if eval_report.recommendation == "ACTIVATE" and args.activate:
        registry.activate(model.metadata().model_version, "APPROVED")
        print(f"🚀 Modelo ACTIVADO")

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


def main() -> int:
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
    train_p.add_argument("--min-positives", type=int, default=10, help="Mínimo pares positivos")
    train_p.add_argument("--min-negatives", type=int, default=10, help="Mínimo pares negativos")
    train_p.add_argument("--test-size", type=float, default=0.2, help="Tamaño test set")
    train_p.add_argument("--val-size", type=float, default=0.1, help="Tamaño validation set")
    train_p.add_argument("--seed", type=int, default=42, help="Random seed")
    train_p.add_argument("--precision-threshold", type=float, default=0.85, help="Umbral precision para activar")
    train_p.add_argument("--recall-threshold", type=float, default=0.70, help="Umbral recall para activar")
    train_p.add_argument("--activate", action="store_true", help="Activar si cumple umbrales")
    train_p.set_defaults(func=cmd_train)

    # evaluate
    eval_p = subparsers.add_parser("evaluate", help="Evaluar modelo existente")
    eval_p.add_argument("model", help="Versión del modelo a evaluar")
    eval_p.add_argument("--activate", action="store_true", help="Activar si recomendación es ACTIVATE")
    eval_p.set_defaults(func=cmd_evaluate)

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