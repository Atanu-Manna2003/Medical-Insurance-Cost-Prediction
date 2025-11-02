import os
import sys
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
from dataclasses import dataclass
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
)
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_models
from src.exception import CustomException
from src.logger import logging
import joblib

# --- DagsHub MLflow Setup ---
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/Atanu-Manna2003/Medical-Insurance-Cost-Prediction.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "Atanu-Manna2003"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "42a97f93ef2d815fdcc70cae2a0929a6fd100526"


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifact', 'model.pkl')
    mlflow_uri: str = os.getenv("MLFLOW_TRACKING_URI", "")
    model_name: str = "MedicalInsuranceRegressor"


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def eval_metrics(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        return r2, mae, rmse

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBoost": XGBRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'poisson']},
                "Random Forest": {'n_estimators': [16, 32, 64, 128]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.05, 0.01], 'n_estimators': [64, 128]},
                "XGBoost": {'learning_rate': [0.1, 0.05, 0.01], 'n_estimators': [64, 128]},
                "CatBoost": {'depth': [6, 8, 10], 'iterations': [50, 100], 'learning_rate': [0.01, 0.1]},
                "AdaBoost": {'n_estimators': [64, 128], 'learning_rate': [0.1, 0.5]},
                "Linear Regression": {}
            }

            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )

            best_model_score = max(model_report.values())
            best_model_name = [k for k, v in model_report.items() if v == best_model_score][0]
            best_model = models[best_model_name]

            logging.info(f"✅ Best Model: {best_model_name} with R2 Score: {best_model_score}")

            # ---- MLflow Logging ----
            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment("Medical Insurance Cost Prediction")

            print(f"🚀 Starting MLflow run for: {best_model_name}")

            with mlflow.start_run(run_name=best_model_name):
                best_model.fit(X_train, y_train)
                preds = best_model.predict(X_test)

                r2, mae, rmse = self.eval_metrics(y_test, preds)

                mlflow.log_param("model_type", best_model_name)
                mlflow.log_params(params.get(best_model_name, {}))
                mlflow.log_metric("r2_score", r2)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)

                # ✅ Save model locally
                model_path = f"artifact/{best_model_name}_model.pkl"
                joblib.dump(best_model, model_path)

                # ✅ Log model artifact (DagsHub-compatible)
                mlflow.log_artifact(model_path)

                import json, os
                os.makedirs("artifact", exist_ok=True)
                metrics = {"r2": r2, "mae": mae, "rmse": rmse}
                with open("artifact/metrics.json", "w") as f:
                  json.dump(metrics, f, indent=2)




            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            return r2

        except Exception as e:
            raise CustomException(e, sys)
