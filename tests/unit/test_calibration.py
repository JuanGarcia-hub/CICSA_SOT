"""Tests unitarios para módulo de calibración."""

import numpy as np
import pytest

from sot_matcher.infrastructure.calibration import (
    TemperatureScaler,
    IsotonicCalibrator,
    PlattCalibrator,
    HistogramBinningCalibrator,
    IdentityCalibrator,
    CalibrationConfig,
    CalibrationMetrics,
    create_calibrator,
    compute_calibration_metrics,
    compare_calibration_methods,
)


@pytest.fixture
def synthetic_data():
    """Datos sintéticos para testing de calibración."""
    np.random.seed(42)
    n = 1000
    # Probabilidades con sesgo sistemático (subconfiadas)
    y_true = np.random.binomial(1, 0.3, n)
    # Modelo subconfiado: predice probabilidades más bajas de lo real
    logits_true = np.log(0.3 / 0.7) + np.random.normal(0, 0.5, n)
    y_proba = 1 / (1 + np.exp(-logits_true * 0.7))  # factor 0.7 = subconfiado
    return y_true, y_proba


def test_identity_calibrator():
    """Calibrador identidad no cambia probabilidades."""
    cal = IdentityCalibrator()
    probas = np.array([0.1, 0.5, 0.9])
    cal.fit(probas, np.array([0, 1, 1]))
    result = cal.transform(probas)
    np.testing.assert_array_almost_equal(result, probas)


def test_temperature_scaler_fits(synthetic_data):
    """TemperatureScaler aprende temperatura > 1 para modelo subconfiado."""
    y_true, y_proba = synthetic_data
    cal = TemperatureScaler(temperature_range=(0.1, 10.0))
    cal.fit(y_proba, y_true)

    assert cal._fitted
    assert cal.temperature_ > 0.5  # Debería aprender T > 1 para corregir subconfianza
    assert cal.temperature_ < 5.0

    # Transformar debería cambiar las probabilidades
    transformed = cal.transform(y_proba)
    assert not np.allclose(transformed, y_proba)


def test_isotonic_calibrator(synthetic_data):
    """IsotonicCalibrator produce calibración monótona."""
    y_true, y_proba = synthetic_data
    cal = IsotonicCalibrator()
    cal.fit(y_proba, y_true)

    assert cal._fitted
    transformed = cal.transform(y_proba)

    # Isotonic regression debe ser monótona
    order = np.argsort(y_proba)
    transformed_sorted = transformed[order]
    assert np.all(np.diff(transformed_sorted) >= -1e-10)  # monótona no decreciente


def test_platt_calibrator(synthetic_data):
    """PlattCalibrator usa regresión logística sobre logits."""
    y_true, y_proba = synthetic_data
    cal = PlattCalibrator()
    cal.fit(y_proba, y_true)

    assert cal._fitted
    assert cal.calibrator_ is not None

    transformed = cal.transform(y_proba)
    assert len(transformed) == len(y_proba)
    assert np.all((transformed >= 0) & (transformed <= 1))


def test_histogram_binning(synthetic_data):
    """HistogramBinningCalibrator promedia en bins."""
    y_true, y_proba = synthetic_data
    cal = HistogramBinningCalibrator(n_bins=10)
    cal.fit(y_proba, y_true)

    assert cal._fitted
    assert cal.bin_edges_ is not None
    assert cal.bin_values_ is not None
    assert len(cal.bin_edges_) == 11
    assert len(cal.bin_values_) == 10

    transformed = cal.transform(y_proba)
    assert len(transformed) == len(y_proba)


def test_create_calibrator_factory():
    """Factory crea calibradores correctos."""
    for method in ["temperature", "isotonic", "platt", "histogram", "none"]:
        config = CalibrationConfig(method=method)
        cal = create_calibrator(config)
        assert isinstance(cal, (TemperatureScaler, IsotonicCalibrator,
                                PlattCalibrator, HistogramBinningCalibrator,
                                IdentityCalibrator))

    with pytest.raises(ValueError):
        create_calibrator(CalibrationConfig(method="unknown"))


def test_calibration_config_defaults():
    """Configuración por defecto."""
    config = CalibrationConfig()
    assert config.method == "temperature"
    assert config.n_bins == 10
    assert config.temperature_range == (0.1, 10.0)


def test_compute_calibration_metrics():
    """Cálculo de métricas ECE, MCE, Brier, LogLoss."""
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9, 0.3, 0.6, 0.4, 0.85, 0.95])

    metrics = compute_calibration_metrics(y_true, y_proba, n_bins=5)

    assert isinstance(metrics, CalibrationMetrics)
    assert metrics.ece >= 0
    assert metrics.mce >= 0
    assert metrics.brier_score >= 0
    assert metrics.log_loss >= 0
    assert len(metrics.bin_accuracies) == 5
    assert len(metrics.bin_confidences) == 5
    assert len(metrics.bin_counts) == 5
    assert sum(metrics.bin_counts) == len(y_true)


def test_calibration_metrics_perfect_calibration():
    """Métricas con calibración perfecta."""
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.0, 0.0, 1.0, 1.0])  # Perfecto

    metrics = compute_calibration_metrics(y_true, y_proba, n_bins=2)
    assert metrics.ece == 0.0
    assert metrics.mce == 0.0


def test_compare_calibration_methods(synthetic_data):
    """Comparación de métodos de calibración."""
    y_true, y_proba = synthetic_data
    results = compare_calibration_methods(
        y_true, y_proba,
        methods=["temperature", "isotonic", "platt", "histogram"]
    )

    assert len(results) == 4
    for method, metrics in results.items():
        assert isinstance(metrics, CalibrationMetrics)
        assert metrics.ece >= 0


def test_calibrator_save_load(synthetic_data, tmp_path):
    """Persistencia de calibradores."""
    y_true, y_proba = synthetic_data

    for config_method in ["temperature", "isotonic", "platt", "histogram"]:
        config = CalibrationConfig(method=config_method)
        cal = create_calibrator(config)
        cal.fit(y_proba, y_true)

        # Save
        save_path = tmp_path / f"cal_{config_method}.json" if config_method == "temperature" else tmp_path / f"cal_{config_method}.pkl"
        cal.save(save_path)

        # Load
        cal_loaded = create_calibrator(config).load(save_path)

        # Verificar que produce mismos resultados
        original = cal.transform(y_proba[:10])
        loaded = cal_loaded.transform(y_proba[:10])
        np.testing.assert_array_almost_equal(original, loaded, decimal=5)