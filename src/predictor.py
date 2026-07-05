"""
ML traffic demand predictor.

This module trains a model to predict near-term vehicle demand by:
- time of day
- intersection
- direction
- weather
- event conditions
- recent demand patterns

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import (
    DEFAULT_TEST_SIZE,
    RF_N_ESTIMATORS,
    RF_MAX_DEPTH,
    RF_MIN_SAMPLES_LEAF,
    RF_RANDOM_STATE,
    ROLLING_WINDOW_SHORT,
    ROLLING_WINDOW_LONG,
)


@dataclass
class ModelEvaluation:
    """
    Stores model evaluation results.
    """

    model_name: str
    mae: float
    rmse: float
    r2: float


class HistoricalAverageDemandModel:
    """
    Simple baseline model.

    It predicts demand using the average vehicle demand for:
    - hour
    - intersection
    - direction

    If a combination is unseen, it falls back to global average demand.
    """

    def __init__(self):
        self.lookup = None
        self.global_average = 0.0

    def fit(self, train_df: pd.DataFrame) -> None:
        self.global_average = float(train_df["target_next_vehicle_demand"].mean())

        self.lookup = (
            train_df
            .groupby(["hour", "intersection_id", "direction"])["target_next_vehicle_demand"]
            .mean()
            .reset_index()
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.lookup is None:
            raise ValueError("Model must be fitted before prediction.")

        merged = X[["hour", "intersection_id", "direction"]].merge(
            self.lookup,
            on=["hour", "intersection_id", "direction"],
            how="left",
        )

        predictions = merged["target_next_vehicle_demand"].fillna(
            self.global_average
        )

        return predictions.to_numpy()


class TrafficDemandPredictor:
    """
    Random forest demand predictor.

    This class handles:
    - feature engineering
    - train/test time split
    - model fitting
    - evaluation
    - feature importance
    """

    def __init__(
            self,
            random_state: int = RF_RANDOM_STATE,
            n_estimators: int = RF_N_ESTIMATORS,
            max_depth: int | None = RF_MAX_DEPTH,
            min_samples_leaf: int = RF_MIN_SAMPLES_LEAF,
    ):
        self.random_state = random_state

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            min_samples_leaf=min_samples_leaf,
        )

        self.feature_columns: List[str] = []

    def prepare_training_data(self, demand_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build ML-ready dataset from generated demand.

        Target:
            next-step vehicle demand by intersection and direction.

        We use a group-wise shift so that NS at intersection 0 predicts
        its own next demand, not another intersection's demand.
        """
        df = demand_df.copy()

        df = df.sort_values(
            ["intersection_id", "direction", "time_step"]
        ).reset_index(drop=True)

        # Direction encoding
        df["direction_is_ns"] = (df["direction"] == "NS").astype(int)

        # Cyclical hour encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Lag and rolling features per intersection-direction pair
        # Lag and rolling features per intersection-direction pair
        group_cols = ["intersection_id", "direction"]

        df["lag_1_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .shift(1)
        )

        df["rolling_mean_3_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .transform(lambda s: s.shift(1).rolling(window=ROLLING_WINDOW_SHORT, min_periods=1).mean())
        )

        df["rolling_mean_5_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .transform(lambda s: s.shift(1).rolling(window=ROLLING_WINDOW_LONG, min_periods=1).mean())
        )

        # Prediction target: next time step demand for same intersection-direction
        df["target_next_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .shift(-1)
        )

        # Drop rows without lag or target
        df = df.dropna(
            subset=[
                "lag_1_vehicle_demand",
                "rolling_mean_3_vehicle_demand",
                "rolling_mean_5_vehicle_demand",
                "target_next_vehicle_demand",
            ]
        ).reset_index(drop=True)

        self.feature_columns = [
            "time_step",
            "hour",
            "hour_sin",
            "hour_cos",
            "intersection_id",
            "direction_is_ns",
            "rain_level",
            "weather_demand_multiplier",
            "weather_capacity_multiplier",
            "event_multiplier",
            "peak_multiplier",
            "intersection_multiplier",
            "direction_multiplier",
            "pedestrian_demand",
            "accident_active",
            "lag_1_vehicle_demand",
            "rolling_mean_3_vehicle_demand",
            "rolling_mean_5_vehicle_demand",
        ]

        return df

    def time_based_train_test_split(
        self,
        model_df: pd.DataFrame,
        test_size: float = 0.2,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by time rather than random rows.

        This avoids training on future observations and testing on the past.
        """
        max_time = model_df["time_step"].max()
        split_time = max_time * (1 - test_size)

        train_df = model_df[model_df["time_step"] <= split_time].copy()
        test_df = model_df[model_df["time_step"] > split_time].copy()

        return train_df, test_df

    def fit(self, train_df: pd.DataFrame) -> None:
        X_train = train_df[self.feature_columns]
        y_train = train_df["target_next_vehicle_demand"]

        self.model.fit(X_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_columns]
        predictions = self.model.predict(X)

        # Demand cannot be negative.
        return np.maximum(predictions, 0)

    def evaluate(self, test_df: pd.DataFrame) -> ModelEvaluation:
        y_true = test_df["target_next_vehicle_demand"].to_numpy()
        y_pred = self.predict(test_df)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        return ModelEvaluation(
            model_name="RandomForestRegressor",
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
        )

    def feature_importance(self) -> pd.DataFrame:
        """
        Return feature importance table.
        """
        importances = self.model.feature_importances_

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": importances,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.reset_index(drop=True)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> ModelEvaluation:
    """
    Shared evaluation helper.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return ModelEvaluation(
        model_name=model_name,
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
    )


def train_and_evaluate_predictor(
    demand_df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RF_RANDOM_STATE,
) -> Dict[str, object]:
    """
    Train and evaluate both:
    - historical average baseline
    - random forest model

    Returns a dictionary with metrics, model, and feature importance.
    """
    predictor = TrafficDemandPredictor(random_state=random_state)

    model_df = predictor.prepare_training_data(demand_df)

    train_df, test_df = predictor.time_based_train_test_split(
        model_df=model_df,
        test_size=test_size,
    )

    # Baseline model
    baseline_model = HistoricalAverageDemandModel()
    baseline_model.fit(train_df)

    baseline_predictions = baseline_model.predict(test_df)
    y_test = test_df["target_next_vehicle_demand"].to_numpy()

    baseline_eval = evaluate_predictions(
        model_name="HistoricalAverage",
        y_true=y_test,
        y_pred=baseline_predictions,
    )

    # ML model
    predictor.fit(train_df)
    model_eval = predictor.evaluate(test_df)

    importance_df = predictor.feature_importance()

    return {
        "predictor": predictor,
        "baseline_model": baseline_model,
        "model_df": model_df,
        "train_df": train_df,
        "test_df": test_df,
        "baseline_evaluation": baseline_eval,
        "model_evaluation": model_eval,
        "feature_importance": importance_df,
    }