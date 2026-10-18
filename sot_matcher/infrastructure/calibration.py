"""Calibración avanzada de probabilidades para clasificadores.

Incluye:
- Temperature Scaling (Platt scaling generalizado)
- Isotonic Regression
- Histogram Binning
- Reliability Diagrams y métricas ECE/MCE
"""

from __future__ import annotations

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV


@dataclass(frozen=True)
class CalibrationMetrics:
    """Métricas de calibración."""
    ece: float          # Expected Calibration Error
    mce: float          # Maximum Calibration Error
    brier_score: float
    log_loss: float
    n_bins: int = 10
    bin_accuracies: list[float] = None
    bin_confidences: list[float] = None
    bin_counts: list[int] = None


@dataclass(frozen=True)
class CalibrationConfig:
    """Configuración de calibración."""
    method: str = "temperature"  # "temperature" | "isotonic" | "platt" | "histogram" | "none"
    n_bins: int = 10
    temperature_range: tuple[float, float] = (0.1, 10.0)


class BaseCalibrator(ABC):
    """Interfaz base para calibradores."""

    @abstractmethod
    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        """Entrena el calibrador con probabilidades y etiquetas."""
        pass

    @abstractmethod
    def transform(self, probas: np.ndarray) -> np.ndarray:
        """Aplica calibración a nuevas probabilidades."""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persiste el calibrador."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "BaseCalibrator":
        """Carga calibrador desde disco."""
        pass


class TemperatureScaler(BaseCalibrator):
    """Temperature Scaling: aprende un parámetro T que escala los logits.

    p_calibrated = sigmoid(logits / T)
    Equivalente a Platt scaling cuando el modelo base es bien calibrado.
    """

    def __init__(self, temperature_range: tuple[float, float] = (0.1, 10.0)):
        self.temperature_range = temperature_range
        self.temperature_: float = 1.0
        self._fitted = False

    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        """Optimiza temperatura minimizando NLL (Negative Log Likelihood)."""
        # Convertir probabilidades a logits
        eps = 1e-15
        probas_clipped = np.clip(probas, eps, 1 - eps)
        logits = np.log(probas_clipped / (1 - probas_clipped))

        def nll(temp):
            scaled_logits = logits / temp
            scaled_probas = 1 / (1 + np.exp(-scaled_logits))
            scaled_probas = np.clip(scaled_probas, eps, 1 - eps)
            return -np.mean(labels * np.log(scaled_probas) + (1 - labels) * np.log(1 - scaled_probas))

        result = minimize_scalar(nll, bounds=self.temperature_range, method='bounded')
        self.temperature_ = result.x
        self._fitted = True

    def transform(self, probas: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probas
        eps = 1e-15
        probas_clipped = np.clip(probas, eps, 1 - eps)
        logits = np.log(probas_clipped / (1 - probas_clipped))
        scaled_logits = logits / self.temperature_
        return 1 / (1 + np.exp(-scaled_logits))

    def save(self, path: Path) -> None:
        data = {"temperature": self.temperature_, "fitted": self._fitted, "type": "temperature"}
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "TemperatureScaler":
        data = json.loads(path.read_text())
        scaler = cls()
        scaler.temperature_ = data["temperature"]
        scaler._fitted = data["fitted"]
        return scaler


class IsotonicCalibrator(BaseCalibrator):
    """Isotonic Regression: calibración no paramétrica monótona."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.calibrator_: Optional[IsotonicRegression] = None
        self._fitted = False

    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        self.calibrator_ = IsotonicRegression(out_of_bounds="clip")
        self.calibrator_.fit(probas, labels)
        self._fitted = True

    def transform(self, probas: np.ndarray) -> np.ndarray:
        if not self._fitted or self.calibrator_ is None:
            return probas
        return self.calibrator_.predict(probas)

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump({"calibrator": self.calibrator_, "fitted": self._fitted, "type": "isotonic"}, f)

    @classmethod
    def load(cls, path: Path) -> "IsotonicCalibrator":
        with path.open("rb") as f:
            data = pickle.load(f)
        calibrator = cls()
        calibrator.calibrator_ = data["calibrator"]
        calibrator._fitted = data["fitted"]
        return calibrator


class PlattCalibrator(BaseCalibrator):
    """Platt Scaling: Logistic Regression sobre logits."""

    def __init__(self):
        self.calibrator_: Optional[LogisticRegression] = None
        self._fitted = False

    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        eps = 1e-15
        probas_clipped = np.clip(probas, eps, 1 - eps)
        logits = np.log(probas_clipped / (1 - probas_clipped)).reshape(-1, 1)

        self.calibrator_ = LogisticRegression(solver="lbfgs", max_iter=1000)
        self.calibrator_.fit(logits, labels)
        self._fitted = True

    def transform(self, probas: np.ndarray) -> np.ndarray:
        if not self._fitted or self.calibrator_ is None:
            return probas
        eps = 1e-15
        probas_clipped = np.clip(probas, eps, 1 - eps)
        logits = np.log(probas_clipped / (1 - probas_clipped)).reshape(-1, 1)
        return self.calibrator_.predict_proba(logits)[:, 1]

    def save(self, path: Path) -> None:
        with path.open("wb") as f:
            pickle.dump({"calibrator": self.calibrator_, "fitted": self._fitted, "type": "platt"}, f)

    @classmethod
    def load(cls, path: Path) -> "PlattCalibrator":
        with path.open("rb") as f:
            data = pickle.load(f)
        calibrator = cls()
        calibrator.calibrator_ = data["calibrator"]
        calibrator._fitted = data["fitted"]
        return calibrator


class HistogramBinningCalibrator(BaseCalibrator):
    """Histogram Binning: promedia probabilidades en bins de confianza."""

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.bin_edges_: Optional[np.ndarray] = None
        self.bin_values_: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        self.bin_values_ = np.zeros(self.n_bins)

        for i in range(self.n_bins):
            mask = (probas >= self.bin_edges_[i]) & (probas < self.bin_edges_[i + 1])
            if i == self.n_bins - 1:  # último bin incluye 1.0
                mask = (probas >= self.bin_edges_[i]) & (probas <= self.bin_edges_[i + 1])

            if mask.any():
                self.bin_values_[i] = labels[mask].mean()
            else:
                # Si bin vacío, interpolar
                self.bin_values_[i] = (self.bin_edges_[i] + self.bin_edges_[i + 1]) / 2

        self._fitted = True

    def transform(self, probas: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return probas

        calibrated = np.zeros_like(probas)
        for i in range(self.n_bins):
            mask = (probas >= self.bin_edges_[i]) & (probas < self.bin_edges_[i + 1])
            if i == self.n_bins - 1:
                mask = (probas >= self.bin_edges_[i]) & (probas <= self.bin_edges_[i + 1])
            calibrated[mask] = self.bin_values_[i]

        return calibrated

    def save(self, path: Path) -> None:
        data = {
            "n_bins": self.n_bins,
            "bin_edges": self.bin_edges_.tolist(),
            "bin_values": self.bin_values_.tolist(),
            "fitted": self._fitted,
            "type": "histogram"
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "HistogramBinningCalibrator":
        data = json.loads(path.read_text())
        calibrator = cls(n_bins=data["n_bins"])
        calibrator.bin_edges_ = np.array(data["bin_edges"])
        calibrator.bin_values_ = np.array(data["bin_values"])
        calibrator._fitted = data["fitted"]
        return calibrator


def create_calibrator(config: CalibrationConfig) -> BaseCalibrator:
    """Factory para crear calibrador según configuración."""
    if config.method == "temperature":
        return TemperatureScaler(temperature_range=config.temperature_range)
    elif config.method == "isotonic":
        return IsotonicCalibrator(n_bins=config.n_bins)
    elif config.method == "platt":
        return PlattCalibrator()
    elif config.method == "histogram":
        return HistogramBinningCalibrator(n_bins=config.n_bins)
    elif config.method == "none":
        return IdentityCalibrator()
    else:
        raise ValueError(f"Método de calibración desconocido: {config.method}")


class IdentityCalibrator(BaseCalibrator):
    """Calibrador identidad (sin calibración)."""

    def fit(self, probas: np.ndarray, labels: np.ndarray) -> None:
        pass

    def transform(self, probas: np.ndarray) -> np.ndarray:
        return probas

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"type": "identity"}))

    @classmethod
    def load(cls, path: Path) -> "IdentityCalibrator":
        return cls()


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> CalibrationMetrics:
    """Calcula métricas de calibración completas incluyendo ECE y MCE."""
    from sklearn.metrics import brier_score_loss, log_loss

    # ECE y MCE
    bin_edges = np.linspace(0, 1, n_bins + 1)

    ece = 0.0
    mce = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        bin_lower = bin_edges[i]
        bin_upper = bin_edges[i + 1]

        # Usar >= lower, < upper para todos los bins excepto el último
        if i == n_bins - 1:
            in_bin = (y_proba >= bin_lower) & (y_proba <= bin_upper)
        else:
            in_bin = (y_proba >= bin_lower) & (y_proba < bin_upper)

        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            mce = max(mce, abs(avg_confidence_in_bin - accuracy_in_bin))
            bin_accuracies.append(float(accuracy_in_bin))
            bin_confidences.append(float(avg_confidence_in_bin))
            bin_counts.append(int(in_bin.sum()))
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)
            bin_counts.append(0)

    brier = float(brier_score_loss(y_true, y_proba))
    logloss = float(log_loss(y_true, y_proba, labels=[0, 1]))

    return CalibrationMetrics(
        ece=float(ece),
        mce=float(mce),
        brier_score=brier,
        log_loss=logloss,
        n_bins=n_bins,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
    title: str = "Reliability Diagram",
    save_path: Optional[Path] = None,
) -> "matplotlib.figure.Figure":
    """Genera diagrama de confiabilidad (requiere matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib requerido para plot_reliability_diagram")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    accuracies = []
    confidences = []
    counts = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        mask = (y_proba >= lower) & (y_proba < upper)
        if i == n_bins - 1:
            mask = (y_proba >= lower) & (y_proba <= upper)

        if mask.any():
            accuracies.append(y_true[mask].mean())
            confidences.append(y_proba[mask].mean())
            counts.append(mask.sum())
        else:
            accuracies.append(0)
            confidences.append(0)
            counts.append(0)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Diagonal perfecta
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.5)

    # Barras de calibración
    bars = ax.bar(bin_centers, accuracies, width=0.9/n_bins, align="center",
                  alpha=0.7, edgecolor="black", label="Model", color="steelblue")

    # Puntos de confianza promedio
    ax.scatter(confidences, accuracies, color="red", s=100, zorder=5,
               label="Avg confidence vs accuracy", edgecolors="black")

    # Tamaño de bins como texto
    for i, (center, acc, conf, cnt) in enumerate(zip(bin_centers, accuracies, confidences, counts)):
        if cnt > 0:
            ax.text(center, acc + 0.02, f"n={cnt}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Mean Predicted Probability (Confidence)")
    ax.set_ylabel("Fraction of Positives (Accuracy)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def compare_calibration_methods(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    methods: list[str] = None,
) -> dict[str, CalibrationMetrics]:
    """Compara múltiples métodos de calibración."""
    if methods is None:
        methods = ["temperature", "isotonic", "platt", "histogram"]

    results = {}
    for method in methods:
        config = CalibrationConfig(method=method)
        calibrator = create_calibrator(config)
        calibrator.fit(y_proba, y_true)
        calibrated = calibrator.transform(y_proba)
        metrics = compute_calibration_metrics(y_true, calibrated)
        results[method] = metrics

    return results