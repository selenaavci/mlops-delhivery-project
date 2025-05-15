import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/Delhivery.csv")

# Select features and target
X = df[[
    "actual_distance_to_destination",
    "osrm_distance",
    "osrm_time",
    "cutoff_factor",
    "factor",
    "segment_factor",
    "segment_osrm_distance"
]]
y = df["actual_time"]

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define best parameters (from Hyperopt results)
best_params = {
    "n_estimators": 290,
    "max_depth": 21,
    "random_state": 42
}

# Start MLflow experiment
with mlflow.start_run(run_name="final_best_rf_model"):
    # Train model
    model = RandomForestRegressor(**best_params)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log parameters and metrics to MLflow
    mlflow.log_param("model_type", "RandomForest-FINAL")
    mlflow.log_param("n_estimators", best_params["n_estimators"])
    mlflow.log_param("max_depth", best_params["max_depth"])
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Log model and register to Model Registry
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Delhivery_RF_Model"
    )

    print(f"✅ Final model MAE: {mae:.2f}")
    print(f"✅ Final model R² Score: {r2:.4f}")
