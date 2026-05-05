import json
import sqlite3
import urllib.request
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
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
# ENSEMBLE LAYER
# -----------------------------
class EnsembleLayer:
    def normalize(self, scores: np.ndarray) -> np.ndarray:
        min_val = scores.min()
        max_val = scores.max()
        if max_val - min_val < 1e-8:
            return np.zeros_like(scores)
        return (scores - min_val) / (max_val - min_val)

    def compute_weights(self, scores_list: List[np.ndarray]) -> List[float]:
        variances = [float(np.var(scores)) for scores in scores_list]
        total = sum(variances)
        if total <= 0:
            return [1.0 / len(variances)] * len(variances)
        return [v / total for v in variances]

    def combine(self, scores_list: List[np.ndarray]) -> Tuple[np.ndarray, List[float]]:
        normalized = [self.normalize(scores) for scores in scores_list]
        weights = self.compute_weights(normalized)
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
        strategy: str = "percentile",
        percentile: float = 95.0,
        std_multiplier: float = 1.5,
    ) -> float:
        if strategy == "percentile":
            return float(np.percentile(final_scores, percentile))
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
    def __init__(self) -> None:
        self.input = InputLayer()
        self.analysis = AnalysisLayer()
        self.optimization = OptimizationLayer()
        self.core = CoreLayer()
        self.ensemble = EnsembleLayer()
        self.post = PostProcessingLayer()
        self.output = OutputLayer()

        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)

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
        threshold_strategy: str = "percentile",
        threshold_percentile: float = 95.0,
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

        final_scores, weights = self.ensemble.combine(scores_list)
        normalized_scores = {
            name: self.ensemble.normalize(info["raw_scores"])
            for name, info in trained_models.items()
        }
        threshold = self.post.threshold(
            final_scores,
            strategy=threshold_strategy,
            percentile=threshold_percentile,
        )
        anomalies = self.post.label(final_scores, threshold)
        report = self.output.report(anomalies, final_scores, weights)
        result_df = self.output.to_dataframe(df, final_scores, anomalies)

        return anomalies, final_scores, {
            "report": report,
            "meta": meta,
            "models": trained_models,
            "model_names": models,
            "model_weights": dict(zip(models, weights)),
            "normalized_model_scores": normalized_scores,
            "threshold": threshold,
            "threshold_strategy": threshold_strategy,
            "threshold_percentile": threshold_percentile,
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
