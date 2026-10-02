import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split, cross_val_score

from preprocessing import get_dataset, prepare_features, FEATURES


def train_models():
    """Train and compare regression models."""

    df = get_dataset()
    X, y = prepare_features(df)

    # Reserve 20% of the data for final testing.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=150,
            max_depth=6,
            random_state=42
        )
    }

    # Compare models using cross-validation on training data.
    cv_scores = {}

    for name, model in models.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            scoring="r2"
        )
        cv_scores[name] = float(scores.mean())

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = models[best_name]

    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    metrics = {
        "R2": r2_score(y_test, predictions),
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(
            mean_squared_error(y_test, predictions)
        )
    }

    results = pd.DataFrame({
        "Actual": y_test.to_numpy(),
        "Predicted": predictions
    })

    return {
        "model": best_model,
        "model_name": best_name,
        "metrics": metrics,
        "cv_scores": cv_scores,
        "results": results,
        "dataset": df
    }


def predict_sales(model, tv, radio, newspaper):
    """Predict sales from advertising budgets."""

    input_data = pd.DataFrame(
        [[tv, radio, newspaper]],
        columns=FEATURES
    )

    prediction = model.predict(input_data)[0]

    return max(0.0, float(prediction))