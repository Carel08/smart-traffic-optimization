"""
ML traffic demand predictor.

This module trains models to predict near-term vehicle demand by:
- time of day
- intersection
- direction
- weather
- event conditions
- recent demand patterns

It compares:
- HistoricalAverage baseline
- Tuned RandomForestRegressor
- Tuned HistGradientBoostingRegressor

The selected model is used to create predicted_vehicle_demand for the
adaptive and Scipy MPC controllers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
    params: Dict[str, Any] = field(default_factory=dict)


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
    Demand predictor wrapper.

    This class handles:
    - feature engineering
    - train/test time split
    - model fitting
    - evaluation
    - feature importance

    model_type can be:
    - "random_forest"
    - "hist_gradient_boosting"
    """

    def __init__(
        self,
        random_state: int = RF_RANDOM_STATE,
        model_type: str = "random_forest",
        model_params: Dict[str, Any] | None = None,
    ):
        self.random_state = random_state
        self.model_type = model_type
        self.model_params = model_params or self._default_params(model_type)
        self.model = self._build_model()
        self.feature_columns: List[str] = []

    @staticmethod
    def _default_params(model_type: str) -> Dict[str, Any]:
        if model_type == "random_forest":
            return {
                "n_estimators": RF_N_ESTIMATORS,
                "max_depth": RF_MAX_DEPTH,
                "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            }

        if model_type == "hist_gradient_boosting":
            return {
                "max_iter": 120,
                "learning_rate": 0.05,
                "max_leaf_nodes": 31,
                "min_samples_leaf": 20,
                "l2_regularization": 0.0,
            }

        raise ValueError(f"Unknown model_type: {model_type}")

    @property
    def model_name(self) -> str:
        if self.model_type == "random_forest":
            return "RandomForestRegressor"
        if self.model_type == "hist_gradient_boosting":
            return "HistGradientBoostingRegressor"
        return self.model_type

    def _build_model(self):
        if self.model_type == "random_forest":
            return RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,
                **self.model_params,
            )

        if self.model_type == "hist_gradient_boosting":
            return HistGradientBoostingRegressor(
                random_state=self.random_state,
                **self.model_params,
            )

        raise ValueError(f"Unknown model_type: {self.model_type}")

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
        group_cols = ["intersection_id", "direction"]

        df["lag_1_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .shift(1)
        )

        df["rolling_mean_3_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .transform(
                lambda s: s.shift(1)
                .rolling(window=ROLLING_WINDOW_SHORT, min_periods=1)
                .mean()
            )
        )

        df["rolling_mean_5_vehicle_demand"] = (
            df
            .groupby(group_cols)["vehicle_demand"]
            .transform(
                lambda s: s.shift(1)
                .rolling(window=ROLLING_WINDOW_LONG, min_periods=1)
                .mean()
            )
        )

        # Prediction target: next time-step demand for same intersection-direction
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
            model_name=self.model_name,
            mae=float(mae),
            rmse=float(rmse),
            r2=float(r2),
            params=self.model_params,
        )

    def feature_importance(
        self,
        evaluation_df: pd.DataFrame | None = None,
        n_repeats: int = 5,
    ) -> pd.DataFrame:
        """
        Return feature importance table.

        Random Forest has native impurity-based importance.
        HistGradientBoostingRegressor does not expose native feature_importances_,
        so we use permutation importance when evaluation data is supplied.
        """
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_

            importance_df = pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": importances,
                }
            ).sort_values("importance", ascending=False)

            return importance_df.reset_index(drop=True)

        if evaluation_df is None or evaluation_df.empty:
            return pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": np.zeros(len(self.feature_columns)),
                }
            )

        X_eval = evaluation_df[self.feature_columns]
        y_eval = evaluation_df["target_next_vehicle_demand"]

        permutation_result = permutation_importance(
            self.model,
            X_eval,
            y_eval,
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
        )

        importance_df = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "importance": permutation_result.importances_mean,
            }
        ).sort_values("importance", ascending=False)

        return importance_df.reset_index(drop=True)


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    params: Dict[str, Any] | None = None,
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
        params=params or {},
    )


def random_forest_tuning_grid() -> List[Dict[str, Any]]:
    """
    Small deterministic tuning grid for Random Forest.
    
    It is intentionally compact so the CLI, Streamlit app, and tests remain fast.
    """
    return [
        {
            "n_estimators": 60,
            "max_depth": 8,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
        {
            "n_estimators": 80,
            "max_depth": 12,
            "min_samples_leaf": 5,
            "max_features": 0.8,
        },
        {
            "n_estimators": 120,
            "max_depth": 12,
            "min_samples_leaf": 3,
            "max_features": 0.8,
        },
        {
            "n_estimators": 120,
            "max_depth": None,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
        },
    ]


def hist_gradient_boosting_tuning_grid() -> List[Dict[str, Any]]:
    """
    Small deterministic tuning grid for HistGradientBoostingRegressor.
    """
    return [
        {
            "max_iter": 100,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.0,
        },
        {
            "max_iter": 150,
            "learning_rate": 0.05,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 20,
            "l2_regularization": 0.01,
        },
        {
            "max_iter": 120,
            "learning_rate": 0.08,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 15,
            "l2_regularization": 0.0,
        },
        {
            "max_iter": 180,
            "learning_rate": 0.03,
            "max_leaf_nodes": 63,
            "min_samples_leaf": 20,
            "l2_regularization": 0.01,
        },
    ]


def _tune_model_family(
    model_type: str,
    param_grid: List[Dict[str, Any]],
    tuning_train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: List[str],
    random_state: int,
) -> tuple[Dict[str, Any], pd.DataFrame]:
    """
    Tune one model family using a time-based validation split.

    The objective is validation RMSE. Lower is better.
    """
    rows = []
    best_params: Dict[str, Any] | None = None
    best_rmse = float("inf")

    for params in param_grid:
        predictor = TrafficDemandPredictor(
            random_state=random_state,
            model_type=model_type,
            model_params=params,
        )
        predictor.feature_columns = feature_columns
        predictor.fit(tuning_train_df)

        evaluation = predictor.evaluate(validation_df)

        row = {
            "model_type": model_type,
            "model_name": evaluation.model_name,
            "validation_mae": evaluation.mae,
            "validation_rmse": evaluation.rmse,
            "validation_r2": evaluation.r2,
            "params": params,
        }
        rows.append(row)

        if evaluation.rmse < best_rmse:
            best_rmse = evaluation.rmse
            best_params = params

    tuning_df = (
        pd.DataFrame(rows)
        .sort_values("validation_rmse", ascending=True)
        .reset_index(drop=True)
    )

    if best_params is None:
        raise ValueError(f"No valid model parameters found for {model_type}.")

    return best_params, tuning_df


def train_and_evaluate_predictor(
    demand_df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    random_state: int = RF_RANDOM_STATE,
    tune_hyperparameters: bool = True,
) -> Dict[str, object]:
    """
    Train and evaluate:
    - historical average baseline
    - tuned Random Forest
    - tuned HistGradientBoostingRegressor

    Model selection is based on a time-based validation split inside the
    training window. Final reported performance is measured on the held-out
    test period.
    """
    feature_builder = TrafficDemandPredictor(random_state=random_state)

    model_df = feature_builder.prepare_training_data(demand_df)

    train_df, test_df = feature_builder.time_based_train_test_split(
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

    # Internal time-based validation split for hyperparameter tuning.
    tuning_train_df, validation_df = feature_builder.time_based_train_test_split(
        model_df=train_df,
        test_size=0.2,
    )

    if tune_hyperparameters:
        rf_best_params, rf_tuning_df = _tune_model_family(
            model_type="random_forest",
            param_grid=random_forest_tuning_grid(),
            tuning_train_df=tuning_train_df,
            validation_df=validation_df,
            feature_columns=feature_builder.feature_columns,
            random_state=random_state,
        )

        hist_best_params, hist_tuning_df = _tune_model_family(
            model_type="hist_gradient_boosting",
            param_grid=hist_gradient_boosting_tuning_grid(),
            tuning_train_df=tuning_train_df,
            validation_df=validation_df,
            feature_columns=feature_builder.feature_columns,
            random_state=random_state,
        )
    else:
        rf_best_params = TrafficDemandPredictor._default_params("random_forest")
        hist_best_params = TrafficDemandPredictor._default_params(
            "hist_gradient_boosting"
        )
        rf_tuning_df = pd.DataFrame()
        hist_tuning_df = pd.DataFrame()

    # Refit best candidate from each family on full training data.
    rf_predictor = TrafficDemandPredictor(
        random_state=random_state,
        model_type="random_forest",
        model_params=rf_best_params,
    )
    rf_predictor.feature_columns = feature_builder.feature_columns
    rf_predictor.fit(train_df)
    rf_eval = rf_predictor.evaluate(test_df)

    hist_predictor = TrafficDemandPredictor(
        random_state=random_state,
        model_type="hist_gradient_boosting",
        model_params=hist_best_params,
    )
    hist_predictor.feature_columns = feature_builder.feature_columns
    hist_predictor.fit(train_df)
    hist_eval = hist_predictor.evaluate(test_df)

    # Select best model using validation RMSE, not test RMSE.
    validation_rows = []
    if not rf_tuning_df.empty:
        validation_rows.append(rf_tuning_df.iloc[0].to_dict())
    else:
        validation_rows.append(
            {
                "model_type": "random_forest",
                "model_name": "RandomForestRegressor",
                "validation_rmse": rf_eval.rmse,
                "params": rf_best_params,
            }
        )

    if not hist_tuning_df.empty:
        validation_rows.append(hist_tuning_df.iloc[0].to_dict())
    else:
        validation_rows.append(
            {
                "model_type": "hist_gradient_boosting",
                "model_name": "HistGradientBoostingRegressor",
                "validation_rmse": hist_eval.rmse,
                "params": hist_best_params,
            }
        )

    validation_summary = pd.DataFrame(validation_rows)
    selected_model_type = validation_summary.sort_values(
        "validation_rmse",
        ascending=True,
    )["model_type"].iloc[0]

    if selected_model_type == "hist_gradient_boosting":
        selected_predictor = hist_predictor
        selected_eval = hist_eval
    else:
        selected_predictor = rf_predictor
        selected_eval = rf_eval

    model_comparison = pd.DataFrame(
        [
            {
                "model": baseline_eval.model_name,
                "MAE": baseline_eval.mae,
                "RMSE": baseline_eval.rmse,
                "R2": baseline_eval.r2,
                "selected": False,
                "params": baseline_eval.params,
            },
            {
                "model": rf_eval.model_name,
                "MAE": rf_eval.mae,
                "RMSE": rf_eval.rmse,
                "R2": rf_eval.r2,
                "selected": selected_model_type == "random_forest",
                "params": rf_eval.params,
            },
            {
                "model": hist_eval.model_name,
                "MAE": hist_eval.mae,
                "RMSE": hist_eval.rmse,
                "R2": hist_eval.r2,
                "selected": selected_model_type == "hist_gradient_boosting",
                "params": hist_eval.params,
            },
        ]
    ).sort_values("RMSE", ascending=True).reset_index(drop=True)

    tuning_results = pd.concat(
        [rf_tuning_df, hist_tuning_df],
        ignore_index=True,
    )

    if not tuning_results.empty:
        tuning_results = tuning_results.sort_values(
            "validation_rmse",
            ascending=True,
        ).reset_index(drop=True)

    importance_df = selected_predictor.feature_importance(
        evaluation_df=test_df,
        n_repeats=5,
    )

    return {
        "predictor": selected_predictor,
        "baseline_model": baseline_model,
        "model_df": model_df,
        "train_df": train_df,
        "test_df": test_df,
        "baseline_evaluation": baseline_eval,
        "model_evaluation": selected_eval,
        "random_forest_evaluation": rf_eval,
        "hist_gradient_boosting_evaluation": hist_eval,
        "model_comparison": model_comparison,
        "tuning_results": tuning_results,
        "selected_model_name": selected_eval.model_name,
        "selected_model_params": selected_eval.params,
        "feature_importance": importance_df,
    }


def add_predictions_to_demand(
    demand_df: pd.DataFrame,
    predictor: TrafficDemandPredictor,
) -> pd.DataFrame:
    """
    Add predicted_vehicle_demand to the demand dataframe.

    The simulator expects predictions at:
        time_step + intersection_id + direction

    For rows where lag features are unavailable, we fall back to current
    vehicle_demand. This keeps the simulation fully runnable.
    """
    demand_with_predictions = demand_df.copy()

    prediction_frame = predictor.prepare_training_data(demand_df)

    if prediction_frame.empty:
        demand_with_predictions["predicted_vehicle_demand"] = (
            demand_with_predictions["vehicle_demand"]
        )
        return demand_with_predictions

    predictions = predictor.predict(prediction_frame)

    prediction_output = prediction_frame[
        ["time_step", "intersection_id", "direction"]
    ].copy()

    prediction_output["predicted_vehicle_demand"] = predictions

    demand_with_predictions = demand_with_predictions.merge(
        prediction_output,
        on=["time_step", "intersection_id", "direction"],
        how="left",
    )

    demand_with_predictions["predicted_vehicle_demand"] = (
        demand_with_predictions["predicted_vehicle_demand"]
        .fillna(demand_with_predictions["vehicle_demand"])
    )

    return demand_with_predictions
