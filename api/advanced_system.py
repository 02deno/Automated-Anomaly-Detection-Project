import json
import sqlite3
import urllib.request
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


# -----------------------------
# INPUT LAYER
# -----------------------------
class InputLayer:
    def load_csv(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)

    def load_api(self, url: str, headers: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        request = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(request) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]

        return pd.DataFrame(payload)

    def load_db(self, db_path: str, query: str) -> pd.DataFrame:
        conn = sqlite3.connect(db_path)
        try:
            return pd.read_sql_query(query, conn)
        finally:
            conn.close()

    def load(self, source: Union[pd.DataFrame, str, Dict[str, Any]]) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if isinstance(source, str) and source.lower().endswith(".csv"):
            return self.load_csv(source)

        if isinstance(source, dict) and source.get("type") == "api":
            return self.load_api(source["url"], headers=source.get("headers"))

        if isinstance(source, dict) and source.get("type") == "db":
            return self.load_db(source["db_path"], source["query"])

        raise ValueError("Unsupported input source type")


# -----------------------------
# ANALYSIS LAYER
# -----------------------------
class AnalysisLayer:
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty:
            raise ValueError("At least one numeric column is required.")

        missing_rate = numeric.isna().mean().mean()
        variances = numeric.var(ddof=0)
        skewness = numeric.skew(numeric_only=True).replace([np.inf, -np.inf], np.nan)
        kurtosis = numeric.kurt(numeric_only=True).replace([np.inf, -np.inf], np.nan)
        zero_rate = (numeric.fillna(0.0) == 0.0).mean().mean()

        corr_abs_mean = 0.0
        high_corr_pair_count = 0
        if numeric.shape[1] >= 2:
            corr = numeric.corr(method="pearson").abs().replace([np.inf, -np.inf], np.nan).fillna(0.0)
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
            if not upper.empty:
                corr_abs_mean = float(upper.mean())
                high_corr_pair_count = int((upper >= 0.85).sum())

        entropy: Dict[str, float] = {}
        for col in numeric.columns:
            s = numeric[col].dropna()
            if s.empty or s.nunique(dropna=True) <= 1:
                entropy[col] = 0.0
                continue
            counts = s.value_counts(normalize=True, bins=min(10, max(2, int(np.sqrt(len(s))))))
            probs = counts.to_numpy(dtype=float)
            entropy[col] = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

        return {
            "samples": numeric.shape[0],
            "features": numeric.shape[1],
            "missing_rate": missing_rate,
            "variance_mean": float(variances.mean()) if not variances.empty else 0.0,
            "feature_variances": variances.to_dict(),
            "skewness_mean_abs": float(skewness.abs().mean()) if not skewness.empty else 0.0,
            "kurtosis_mean_abs": float(kurtosis.abs().mean()) if not kurtosis.empty else 0.0,
            "feature_skewness": skewness.fillna(0.0).to_dict(),
            "feature_kurtosis": kurtosis.fillna(0.0).to_dict(),
            "sparsity_zero_rate": float(zero_rate) if not pd.isna(zero_rate) else 0.0,
            "correlation_abs_mean": corr_abs_mean,
            "high_corr_pair_count": high_corr_pair_count,
            "feature_entropy": entropy,
            "numeric_columns": numeric.columns.tolist(),
        }

    def select_models(self, meta: Dict[str, Any]) -> List[str]:
        models = ["iforest", "ocsvm"]

        if meta["samples"] >= 10:
            models.append("lof")

        if meta["samples"] > 50 and meta["features"] >= 5:
            models.append("autoencoder")

        if meta["samples"] > 200 and meta["features"] >= 3:
            models.append("lstm")

        return models


# -----------------------------
# OPTIMIZATION LAYER
# -----------------------------
class OptimizationLayer:
    def optimize_iforest(self, X: np.ndarray) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            n_estimators = trial.suggest_int("n_estimators", 50, 200)
            max_samples = trial.suggest_float("max_samples", 0.5, 1.0)
            model = IsolationForest(n_estimators=n_estimators, max_samples=max_samples, random_state=42)
            model.fit(X)
            scores = -model.score_samples(X)
            return float(np.std(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=8, show_progress_bar=False)
        return study.best_params

    def optimize_ocsvm(self, X: np.ndarray) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            nu = trial.suggest_float("nu", 0.001, 0.2)
            gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
            model = OneClassSVM(nu=nu, gamma=gamma)
            model.fit(X)
            scores = -model.score_samples(X)
            return float(np.std(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=8, show_progress_bar=False)
        return study.best_params

    def optimize_lof(self, X: np.ndarray) -> Dict[str, Any]:
        n = X.shape[0]
        if n < 3:
            return {"n_neighbors": 2, "contamination": "auto"}

        def objective(trial: optuna.Trial) -> float:
            high = max(2, min(35, n - 1))
            n_neighbors = trial.suggest_int("n_neighbors", 2, high)
            metric = trial.suggest_categorical("metric", ["minkowski", "euclidean"])
            model = LocalOutlierFactor(n_neighbors=n_neighbors, metric=metric, contamination="auto")
            model.fit_predict(X)
            scores = -model.negative_outlier_factor_
            return float(np.std(scores))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=8, show_progress_bar=False)
        return study.best_params | {"contamination": "auto"}

    def optimize_autoencoder(self, X: np.ndarray) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            hidden_dim = trial.suggest_int("hidden_dim", 4, max(8, X.shape[1] // 2))
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            model = Autoencoder(X.shape[1], hidden_dim)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            X_tensor = torch.tensor(X, dtype=torch.float32)
            for _ in range(10):
                optimizer.zero_grad()
                recon = model(X_tensor)
                loss = loss_fn(recon, X_tensor)
                loss.backward()
                optimizer.step()
            return float(loss.detach().item())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=8, show_progress_bar=False)
        return study.best_params

    def optimize_lstm(self, X: np.ndarray) -> Dict[str, Any]:
        def objective(trial: optuna.Trial) -> float:
            hidden_size = trial.suggest_int("hidden_size", 8, 32)
            lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
            seq_len = trial.suggest_int("seq_len", 5, min(15, max(5, X.shape[0] // 5)))
            model = LSTMAutoencoder(X.shape[1], hidden_size)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            X_seq = CoreLayer.build_sequence_windows(X, seq_len)
            X_tensor = torch.tensor(X_seq, dtype=torch.float32)
            for _ in range(6):
                optimizer.zero_grad()
                recon = model(X_tensor)
                loss = loss_fn(recon, X_tensor)
                loss.backward()
                optimizer.step()
            return float(loss.detach().item())

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=6, show_progress_bar=False)
        return study.best_params

    def optimize(self, model_name: str, X: np.ndarray) -> Dict[str, Any]:
        if model_name == "iforest":
            return self.optimize_iforest(X)
        if model_name == "ocsvm":
            return self.optimize_ocsvm(X)
        if model_name == "lof":
            return self.optimize_lof(X)
        if model_name == "autoencoder":
            return self.optimize_autoencoder(X)
        if model_name == "lstm":
            return self.optimize_lstm(X)
        return {}


# -----------------------------
# DEEP MODELS
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(hidden_dim * 2, 16)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim * 2, 16), hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim * 2, 16)),
            nn.ReLU(),
            nn.Linear(max(hidden_dim * 2, 16), input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, num_layers: int = 1):
        super().__init__()
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=input_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder(x)
        decoded, _ = self.decoder(encoded)
        return decoded


# -----------------------------
# CORE PROCESSING LAYER
# -----------------------------
class CoreLayer:
    @staticmethod
    def build_sequence_windows(X: np.ndarray, seq_len: int = 10) -> np.ndarray:
        if X.shape[0] < seq_len:
            return X.reshape((1, X.shape[0], X.shape[1]))
        windows = []
        for start in range(0, X.shape[0] - seq_len + 1):
            windows.append(X[start : start + seq_len])
        return np.stack(windows)

    def train_iforest(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        model = IsolationForest(**params, random_state=42)
        model.fit(X)
        return model, -model.score_samples(X)

    def train_ocsvm(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        model = OneClassSVM(**params)
        model.fit(X)
        return model, -model.score_samples(X)

    def train_lof(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        model = LocalOutlierFactor(**params)
        model.fit_predict(X)
        return model, -model.negative_outlier_factor_

    def train_autoencoder(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        hidden_dim = params.get("hidden_dim", max(8, X.shape[1] // 2))
        lr = params.get("lr", 1e-3)
        model = Autoencoder(X.shape[1], hidden_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        X_tensor = torch.tensor(X, dtype=torch.float32)

        for _ in range(20):
            optimizer.zero_grad()
            recon = model(X_tensor)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recon = model(X_tensor)
            scores = torch.mean((X_tensor - recon) ** 2, dim=1).cpu().numpy()

        return model, scores

    def train_lstm(self, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        seq_len = params.get("seq_len", min(10, max(5, X.shape[0] // 10)))
        hidden_size = params.get("hidden_size", 16)
        lr = params.get("lr", 1e-3)
        num_layers = params.get("num_layers", 1)
        X_seq = self.build_sequence_windows(X, seq_len)
        X_tensor = torch.tensor(X_seq, dtype=torch.float32)

        model = LSTMAutoencoder(X.shape[1], hidden_size, num_layers)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(15):
            optimizer.zero_grad()
            recon = model(X_tensor)
            loss = loss_fn(recon, X_tensor)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            recon = model(X_tensor)
            seq_scores = torch.mean((X_tensor - recon) ** 2, dim=(1, 2)).cpu().numpy()

        if seq_scores.shape[0] == 1:
            scores = np.repeat(seq_scores, X.shape[0])
        else:
            scores = np.zeros(X.shape[0], dtype=float)
            count = np.zeros_like(scores)
            for idx in range(seq_scores.shape[0]):
                scores[idx : idx + seq_len] += seq_scores[idx]
                count[idx : idx + seq_len] += 1
            scores = scores / np.maximum(count, 1)

        return model, scores

    def train(self, model_name: str, X: np.ndarray, params: Dict[str, Any]) -> Tuple[Any, np.ndarray]:
        if model_name == "iforest":
            return self.train_iforest(X, params)
        if model_name == "ocsvm":
            return self.train_ocsvm(X, params)
        if model_name == "lof":
            return self.train_lof(X, params)
        if model_name == "autoencoder":
            return self.train_autoencoder(X, params)
        if model_name == "lstm":
            return self.train_lstm(X, params)
        raise ValueError(f"Unsupported model {model_name}")


# -----------------------------
# DOMAIN DETECTION LAYER
# -----------------------------
class DomainDetectionLayer:
    def _numeric_matrix(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        numeric = df.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            return numeric, np.empty((len(df), 0), dtype=float)
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(numeric.median(numeric_only=True)).fillna(0.0)
        return numeric, numeric.to_numpy(dtype=float)

    def flatline_scores(self, df: pd.DataFrame) -> np.ndarray:
        numeric, X = self._numeric_matrix(df)
        if X.shape[1] == 0:
            return np.zeros(len(df), dtype=float)

        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        std = np.nanstd(X, axis=0)
        scale = np.where(mad > 1e-8, 1.4826 * mad, np.where(std > 1e-8, std, 1.0))
        robust_z = np.abs((X - med) / scale)

        # Dead/stuck sensors often create rows that are implausibly close to a
        # sensor's central calibration value across many columns at once.
        median_centrality = np.exp(-6.0 * robust_z).mean(axis=1)
        near_median_ratio = (robust_z <= 0.08).mean(axis=1)

        repeated_cols = np.zeros_like(X, dtype=float)
        for col_idx, col in enumerate(numeric.columns):
            rounded = pd.Series(np.round(X[:, col_idx], 10), index=numeric.index)
            counts = rounded.map(rounded.value_counts()).to_numpy(dtype=float)
            repeated_cols[:, col_idx] = np.maximum(0.0, counts - 1.0) / max(len(rounded) - 1, 1)
        repeated_ratio = repeated_cols.mean(axis=1)

        return (0.65 * median_centrality) + (0.25 * near_median_ratio) + (0.10 * repeated_ratio)

    def temporal_change_scores(self, df: pd.DataFrame) -> np.ndarray:
        _, X = self._numeric_matrix(df)
        if X.shape[0] < 3 or X.shape[1] == 0:
            return np.zeros(len(df), dtype=float)

        diff_prev = np.vstack([np.zeros((1, X.shape[1])), np.abs(np.diff(X, axis=0))])
        diff_next = np.vstack([np.abs(np.diff(X, axis=0)), np.zeros((1, X.shape[1]))])
        local_change = np.maximum(diff_prev, diff_next)

        med = np.nanmedian(local_change, axis=0)
        mad = np.nanmedian(np.abs(local_change - med), axis=0)
        std = np.nanstd(local_change, axis=0)
        scale = np.where(mad > 1e-8, 1.4826 * mad, np.where(std > 1e-8, std, 1.0))
        robust_z = local_change / scale
        return np.nanmean(robust_z, axis=1)

    def freeze_scores(self, df: pd.DataFrame, window: int = 4) -> np.ndarray:
        _, X = self._numeric_matrix(df)
        if X.shape[0] < 3 or X.shape[1] == 0:
            return np.zeros(len(df), dtype=float)

        diffs = np.abs(np.diff(X, axis=0))
        scale = np.nanmedian(diffs, axis=0)
        fallback = np.nanstd(X, axis=0) * 0.01
        eps = np.maximum(np.where(scale > 1e-8, scale * 0.05, fallback), 1e-8)
        same_as_previous = diffs <= eps

        per_row = np.vstack([np.zeros((1, X.shape[1]), dtype=float), same_as_previous.astype(float)])
        streak = np.zeros_like(per_row, dtype=float)
        for row in range(1, per_row.shape[0]):
            streak[row] = np.where(per_row[row] > 0, streak[row - 1] + 1.0, 0.0)

        rolling = pd.DataFrame(per_row).rolling(window=window, min_periods=1).mean().to_numpy()
        freeze_ratio = rolling.mean(axis=1)
        streak_strength = np.clip(streak.max(axis=1) / float(max(window, 1)), 0.0, 1.0)
        multi_column_freeze = (rolling >= 0.75).mean(axis=1)
        return (0.45 * freeze_ratio) + (0.35 * streak_strength) + (0.20 * multi_column_freeze)

    def score(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {
            "flatline": self.flatline_scores(df),
            "temporal_change": self.temporal_change_scores(df),
            "freeze": self.freeze_scores(df),
        }


# -----------------------------
# ENSEMBLE LAYER
# -----------------------------
class EnsembleLayer:
    def __init__(self, calibrated_weights: Optional[Dict[str, float]] = None) -> None:
        self.calibrated_weights = calibrated_weights or {}

    def normalize(self, scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if scores.size == 0:
            return scores
        scores = np.nan_to_num(scores, nan=0.0, posinf=np.nanmax(scores[np.isfinite(scores)]) if np.isfinite(scores).any() else 0.0, neginf=0.0)
        low = float(np.percentile(scores, 1))
        high = float(np.percentile(scores, 99))
        if high > low:
            scores = np.clip(scores, low, high)
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def compute_weights(self, scores_list: List[np.ndarray], names: Optional[List[str]] = None) -> List[float]:
        variances = [float(np.var(scores)) for scores in scores_list]
        if names:
            base_weights = {
                "iforest": 0.45,
                "lof": 0.45,
                "ocsvm": 0.04,
                "autoencoder": 0.20,
                "lstm": 0.15,
                "temporal_change": 0.06,
                "flatline": 0.65,
                "freeze": 0.55,
            }
            base_weights.update({k: float(v) for k, v in self.calibrated_weights.items()})
            weights = np.asarray(
                [base_weights.get(name, 0.10) * max(0.05, variances[idx]) for idx, name in enumerate(names)],
                dtype=float,
            )
            if "flatline" in names:
                for idx, name in enumerate(names):
                    if name != "flatline":
                        weights[idx] *= 0.55
        else:
            weights = np.asarray(variances, dtype=float)
        total = float(np.sum(weights))
        if total <= 0:
            return [1.0 / len(variances)] * len(variances)
        return (weights / total).tolist()

    def combine(self, scores_list: List[np.ndarray], names: Optional[List[str]] = None) -> Tuple[np.ndarray, List[float]]:
        normalized = [self.normalize(scores) for scores in scores_list]
        weights = self.compute_weights(normalized, names=names)
        final = np.zeros_like(normalized[0], dtype=float)
        for weight, score in zip(weights, normalized):
            final += weight * score
        return final, weights


# -----------------------------
# POST PROCESSING LAYER
# -----------------------------
class PostProcessingLayer:
    def threshold(
        self,
        final_scores: np.ndarray,
        strategy: str = "adaptive_gap",
        percentile: float = 95.0,
        std_multiplier: float = 1.5,
        expected_contamination: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> float:
        if strategy == "percentile":
            return float(np.percentile(final_scores, percentile))
        if strategy == "expected_contamination":
            contamination = 0.05 if expected_contamination is None else float(expected_contamination)
            contamination = min(max(contamination, 1.0 / max(len(final_scores), 1)), 0.5)
            threshold = float(np.percentile(final_scores, 100.0 * (1.0 - contamination)))
            return float(np.nextafter(threshold, -np.inf))
        if strategy == "top_k":
            scores = np.asarray(final_scores, dtype=float)
            k = 1 if top_k is None else max(1, min(int(top_k), scores.size))
            ordered = np.sort(scores)
            return float(np.nextafter(float(ordered[-k]), -np.inf))
        if strategy == "adaptive_gap":
            scores = np.asarray(final_scores, dtype=float)
            if scores.size < 4 or float(np.max(scores) - np.min(scores)) < 1e-8:
                return float(np.percentile(scores, percentile))
            ordered = np.sort(scores)
            max_anomaly_count = max(1, min(int(np.ceil(scores.size * 0.20)), scores.size - 1))
            best_gap = -1.0
            best_threshold = float(np.percentile(scores, percentile))
            score_range = float(ordered[-1] - ordered[0]) or 1.0
            for count in range(1, max_anomaly_count + 1):
                split = scores.size - count
                gap = float(ordered[split] - ordered[split - 1])
                normalized_gap = gap / score_range
                if normalized_gap > best_gap:
                    best_gap = normalized_gap
                    best_threshold = float((ordered[split] + ordered[split - 1]) / 2.0)
            if best_gap >= 0.08:
                return best_threshold
            return float(np.percentile(scores, percentile))
        if strategy == "mean_std":
            mean = float(np.mean(final_scores))
            std = float(np.std(final_scores))
            return mean + std_multiplier * std
        mean = float(np.mean(final_scores))
        std = float(np.std(final_scores))
        return mean + 1.5 * std

    def label(self, final_scores: np.ndarray, threshold: float) -> np.ndarray:
        return final_scores > threshold

    def best_percentile_threshold(
        self,
        final_scores: np.ndarray,
        y_true: np.ndarray,
        percentiles: List[float] | None = None,
    ) -> Dict[str, Any]:
        percentiles = percentiles or [float(p) for p in range(50, 100)]
        yt = np.asarray(y_true).astype(int).ravel()
        best = {"percentile": 95.0, "threshold": self.threshold(final_scores), "f1": 0.0}
        for percentile in percentiles:
            threshold = self.threshold(final_scores, strategy="percentile", percentile=percentile)
            pred = self.label(final_scores, threshold).astype(int)
            tp = float(np.sum((yt == 1) & (pred == 1)))
            fp = float(np.sum((yt == 0) & (pred == 1)))
            fn = float(np.sum((yt == 1) & (pred == 0)))
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            if f1 > best["f1"]:
                best = {"percentile": percentile, "threshold": threshold, "f1": float(f1)}
        return best


# -----------------------------
# META SELECTION LAYER
# -----------------------------
class MetaSelectionLayer:
    def __init__(self, profiles: Optional[List[Dict[str, Any]]] = None) -> None:
        self.profiles = profiles or []
        self.allowed_sources = self._allowed_sources()
        self.learned_profiles = [p for p in self.profiles if isinstance(p.get("feature_vector"), list)]
        self._classifier: Optional[RandomForestClassifier] = None
        self._scaler: Optional[StandardScaler] = None
        self._train_learned_selector()

    def choose(
        self,
        meta: Dict[str, Any],
        available_sources: List[str],
        normalized_scores: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, Any]:
        if not self.profiles:
            return {"enabled": False, "selected_source": None}
        if self._classifier is not None and normalized_scores is not None:
            learned = self._choose_learned(meta, available_sources, normalized_scores)
            if learned.get("enabled"):
                return learned

        current = self._vector(meta)
        best_profile: Optional[Dict[str, Any]] = None
        best_distance = float("inf")
        for profile in self.profiles:
            source = str(profile.get("selected_source") or "")
            if source not in available_sources:
                continue
            distance = float(np.linalg.norm(current - self._vector(profile)))
            if distance < best_distance:
                best_distance = distance
                best_profile = profile

        if best_profile is None:
            return {"enabled": False, "selected_source": None}

        return {
            "enabled": True,
            "selected_source": str(best_profile.get("selected_source")),
            "matched_dataset": best_profile.get("dataset"),
            "distance": best_distance,
            "expected_contamination": best_profile.get("expected_contamination"),
            "threshold_strategy": best_profile.get("threshold_strategy", "expected_contamination"),
            "selector_mode": "nearest_profile",
        }

    def _vector(self, meta: Dict[str, Any]) -> np.ndarray:
        samples = max(float(meta.get("samples", meta.get("sample_count", 1.0))), 1.0)
        features = max(float(meta.get("features", meta.get("feature_count", 1.0))), 1.0)
        return np.asarray(
            [
                np.log10(samples),
                np.log10(features),
                float(meta.get("missing_rate", 0.0)),
                float(meta.get("sparsity_zero_rate", 0.0)),
                float(meta.get("correlation_abs_mean", 0.0)),
                float(meta.get("high_corr_pair_count", 0.0)) / max(features * features, 1.0),
            ],
            dtype=float,
        )

    def _allowed_sources(self) -> List[str]:
        for profile in self.profiles:
            sources = profile.get("_allowed_sources")
            if isinstance(sources, list) and sources:
                return [str(source) for source in sources]
        sources = sorted({str(p.get("selected_source")) for p in self.profiles if p.get("selected_source")})
        return sources or ["ensemble", "iforest", "ocsvm", "lof", "temporal_change"]

    def _train_learned_selector(self) -> None:
        rows: List[List[float]] = []
        labels: List[str] = []
        for profile in self.learned_profiles:
            source = str(profile.get("selected_source") or "")
            vector = profile.get("feature_vector")
            if source and isinstance(vector, list):
                rows.append([float(v) for v in vector])
                labels.append(source)
        if len(rows) < 2 or len(set(labels)) < 2:
            return
        X = np.asarray(rows, dtype=float)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._classifier = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
        )
        self._classifier.fit(X_scaled, labels)

    def feature_vector(
        self,
        meta: Dict[str, Any],
        normalized_scores: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[float]:
        vector = self._vector(meta).tolist()
        scores_by_source = normalized_scores or {}
        for source in self.allowed_sources:
            vector.extend(self._score_stats(scores_by_source.get(source)))
        return [float(v) for v in vector]

    def _score_stats(self, scores: Optional[np.ndarray]) -> List[float]:
        if scores is None:
            return [0.0] * 12
        arr = np.asarray(scores, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return [0.0] * 12
        if float(np.max(arr) - np.min(arr)) < 1e-12:
            return [
                1.0,
                float(np.mean(arr)),
                0.0,
                float(np.median(arr)),
                0.0,
                0.0,
                0.0,
                float(np.max(arr)),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        centered = arr - mean
        skew = float(np.mean(centered**3) / max(std**3, 1e-12))
        p50, p75, p90, p95, p99 = [float(np.percentile(arr, p)) for p in (50, 75, 90, 95, 99)]
        top_count = max(1, int(np.ceil(arr.size * 0.01)))
        ordered = np.sort(arr)
        top_mean = float(np.mean(ordered[-top_count:]))
        rest_mean = float(np.mean(ordered[:-top_count])) if ordered.size > top_count else mean
        return [
            1.0,
            mean,
            std,
            p50,
            p90,
            p95,
            p99,
            float(np.max(arr)),
            p75 - float(np.percentile(arr, 25)),
            float(np.max(arr) - p99),
            top_mean - rest_mean,
            skew,
        ]

    def _choose_learned(
        self,
        meta: Dict[str, Any],
        available_sources: List[str],
        normalized_scores: Dict[str, np.ndarray],
    ) -> Dict[str, Any]:
        if self._classifier is None or self._scaler is None:
            return {"enabled": False, "selected_source": None}
        current = np.asarray([self.feature_vector(meta, normalized_scores)], dtype=float)
        current_scaled = self._scaler.transform(current)
        classes = [str(c) for c in self._classifier.classes_]
        probabilities = self._classifier.predict_proba(current_scaled)[0]
        ranked = sorted(zip(classes, probabilities), key=lambda item: float(item[1]), reverse=True)
        selected_source = next((source for source, _ in ranked if source in available_sources), None)
        if selected_source is None:
            return {"enabled": False, "selected_source": None}

        best_profile = None
        best_distance = float("inf")
        current_vec = current[0]
        for profile in self.learned_profiles:
            if str(profile.get("selected_source")) != selected_source:
                continue
            vector = np.asarray(profile.get("feature_vector", []), dtype=float)
            if vector.shape != current_vec.shape:
                continue
            distance = float(np.linalg.norm(current_vec - vector))
            if distance < best_distance:
                best_distance = distance
                best_profile = profile
        if best_profile is None:
            best_profile = next((p for p in self.learned_profiles if str(p.get("selected_source")) == selected_source), {})

        return {
            "enabled": True,
            "selected_source": selected_source,
            "matched_dataset": best_profile.get("dataset"),
            "distance": best_distance,
            "expected_contamination": best_profile.get("expected_contamination"),
            "threshold_strategy": best_profile.get("threshold_strategy", "expected_contamination"),
            "selector_mode": "learned_score_diagnostics",
            "class_probabilities": {source: round(float(prob), 6) for source, prob in ranked},
        }


# -----------------------------
# OUTPUT / REPORTING LAYER
# -----------------------------
class OutputLayer:
    def report(self, anomalies: np.ndarray, final_scores: np.ndarray, weights: List[float]) -> Dict[str, Any]:
        report = {
            "total_samples": int(final_scores.shape[0]),
            "anomaly_count": int(np.sum(anomalies)),
            "anomaly_ratio": float(np.mean(anomalies)),
            "model_weights": weights,
            "top_scores": final_scores.argsort()[-10:][::-1].tolist(),
        }
        print("[REPORT] anomaly_count=", report["anomaly_count"], "weights=", weights)
        return report

    def to_dataframe(self, df: pd.DataFrame, final_scores: np.ndarray, anomalies: np.ndarray) -> pd.DataFrame:
        result = df.copy()
        result["anomaly_score"] = final_scores
        result["is_anomaly"] = anomalies.astype(int)
        return result


# -----------------------------
# ADVANCED SYSTEM
# -----------------------------
class AdvancedAnomalySystem:
    def __init__(
        self,
        weights_config_path: Optional[Union[str, Path]] = None,
        meta_config_path: Optional[Union[str, Path]] = None,
    ) -> None:
        calibrated_weights = self._load_calibrated_weights(weights_config_path)
        meta_profiles = self._load_meta_profiles(meta_config_path)
        self.input = InputLayer()
        self.analysis = AnalysisLayer()
        self.optimization = OptimizationLayer()
        self.core = CoreLayer()
        self.domain = DomainDetectionLayer()
        self.ensemble = EnsembleLayer(calibrated_weights=calibrated_weights)
        self.meta_selector = MetaSelectionLayer(meta_profiles)
        self.post = PostProcessingLayer()
        self.output = OutputLayer()

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

    def _load_calibrated_weights(self, weights_config_path: Optional[Union[str, Path]]) -> Dict[str, float]:
        configured = weights_config_path or os.environ.get("AUTOAD_WEIGHTS_CONFIG")
        if not configured:
            return {}
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.exists():
            return {}
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                import yaml  # type: ignore

                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        weights = data.get("weights", data) if isinstance(data, dict) else {}
        return {str(k): float(v) for k, v in weights.items() if isinstance(v, (int, float))}

    def _load_meta_profiles(self, meta_config_path: Optional[Union[str, Path]]) -> List[Dict[str, Any]]:
        configured = meta_config_path or os.environ.get("AUTOAD_META_CONFIG")
        if not configured:
            return []
        path = Path(configured)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        if not path.exists():
            return []
        try:
            if path.suffix.lower() in {".yaml", ".yml"}:
                import yaml  # type: ignore

                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            else:
                data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        if isinstance(data, dict):
            profiles = data.get("profiles", data)
            allowed_sources = data.get("allowed_sources")
            selector_mode = data.get("selector_mode")
        else:
            profiles = data
            allowed_sources = None
            selector_mode = None
        out = [p for p in profiles if isinstance(p, dict)]
        for profile in out:
            if isinstance(allowed_sources, list):
                profile["_allowed_sources"] = allowed_sources
            if selector_mode:
                profile["_selector_mode"] = selector_mode
        return out

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        numeric = df.select_dtypes(include=[np.number]).fillna(0.0)
        X = numeric.values.astype(float)
        X_scaled = self.scaler.fit_transform(X)
        if X_scaled.shape[1] > 1:
            X_reduced = self.pca.fit_transform(X_scaled)
            return X_reduced
        return X_scaled

    def run(
        self,
        source: Union[pd.DataFrame, str, Dict[str, Any]],
        *,
        threshold_strategy: str = "adaptive_gap",
        threshold_percentile: float = 95.0,
        expected_contamination: Optional[float] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        df = self.input.load(source)
        X = self.preprocess(df)

        meta = self.analysis.analyze(df)
        models = self.analysis.select_models(meta)

        scores_list: List[np.ndarray] = []
        trained_models: Dict[str, Any] = {}

        for model_name in models:
            params = self.optimization.optimize(model_name, X)
            trained, scores = self.core.train(model_name, X, params)
            scores_list.append(scores)
            trained_models[model_name] = {
                "object": trained,
                "params": params,
                "raw_scores": scores,
            }

        domain_scores = self.domain.score(df)
        for detector_name, detector_scores in domain_scores.items():
            trained_models[detector_name] = {
                "object": None,
                "params": {},
                "raw_scores": detector_scores,
            }

        ensemble_names = list(models)
        flatline_scores = domain_scores.get("flatline")
        if flatline_scores is not None and flatline_scores.size:
            sorted_flatline = np.sort(np.asarray(flatline_scores, dtype=float))
            second_best = float(sorted_flatline[-2]) if sorted_flatline.size >= 2 else float(sorted_flatline[-1])
            if second_best >= 0.80:
                scores_list.append(flatline_scores)
                ensemble_names.append("flatline")

        temporal_scores = domain_scores.get("temporal_change")
        if temporal_scores is not None and temporal_scores.size and len(df) >= 12:
            scores_list.append(temporal_scores)
            ensemble_names.append("temporal_change")

        freeze_scores = domain_scores.get("freeze")
        if freeze_scores is not None and freeze_scores.size and float(np.percentile(freeze_scores, 95)) >= 0.35:
            scores_list.append(freeze_scores)
            ensemble_names.append("freeze")

        score_names = list(trained_models.keys())
        ensemble_scores, weights = self.ensemble.combine(scores_list, names=ensemble_names)
        normalized_scores = {
            name: self.ensemble.normalize(info["raw_scores"])
            for name, info in trained_models.items()
        }
        normalized_scores["ensemble"] = self.ensemble.normalize(ensemble_scores)
        final_scores = ensemble_scores
        meta_selection = self.meta_selector.choose(meta, list(normalized_scores.keys()), normalized_scores)
        if meta_selection.get("enabled"):
            selected_source = str(meta_selection["selected_source"])
            final_scores = normalized_scores[selected_source]
            if expected_contamination is None and meta_selection.get("expected_contamination") is not None:
                expected_contamination = float(meta_selection["expected_contamination"])
            if threshold_strategy == "adaptive_gap":
                threshold_strategy = str(meta_selection.get("threshold_strategy") or "expected_contamination")
        threshold = self.post.threshold(
            final_scores,
            strategy=threshold_strategy,
            percentile=threshold_percentile,
            expected_contamination=expected_contamination,
            top_k=top_k,
        )
        anomalies = self.post.label(final_scores, threshold)
        report = self.output.report(anomalies, final_scores, weights)
        result_df = self.output.to_dataframe(df, final_scores, anomalies)

        return anomalies, final_scores, {
            "report": report,
            "meta": meta,
            "models": trained_models,
            "model_names": score_names,
            "model_weights": dict(zip(ensemble_names, weights)),
            "ensemble_score_sources": ensemble_names,
            "normalized_model_scores": normalized_scores,
            "meta_selection": meta_selection,
            "threshold": threshold,
            "threshold_strategy": threshold_strategy,
            "threshold_percentile": threshold_percentile,
            "expected_contamination": expected_contamination,
            "top_k": top_k,
            "results": result_df,
        }


if __name__ == "__main__":
    system = AdvancedAnomalySystem()
    sample_df = pd.DataFrame(
        np.random.randn(300, 6),
        columns=[f"feature_{i}" for i in range(6)],
    )
    anomalies, scores, details = system.run(sample_df)
    print(details["report"])
