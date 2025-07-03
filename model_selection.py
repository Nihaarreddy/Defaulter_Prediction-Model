import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import mlflow
import time

from evidently import Report
from evidently.presets import DataDriftPreset
 
def evaluate_models_with_grid_search(X_train, X_test, y_train, y_test,new_df):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            "params": {
                "C": [0.01, 0.1, 1, 10]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        }
    }
 
    best_score = 0
    best_overall_model = None
    best_overall_params = None
    best_run_id = None
    best_model_name = None
 
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Loan_Default_Classification")
 
    for name, item in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\n Training {name} with GridSearchCV...")
 
            grid = GridSearchCV(item["model"], item["params"], cv=3, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)
 
            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)
 
            # Metrics
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
 
            # Log to MLflow
            mlflow.log_params(grid.best_params_)
            mlflow.log_metrics({
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1_score": f1
            })
 
            mlflow.sklearn.log_model(best_model, artifact_path="model")
            mlflow.set_tag("model_name", name)
 
            print(f"\n Best Params for {name}: {grid.best_params_}")
            print(f" Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
            print(f" Classification Report:\n{classification_report(y_test, preds)}")
            print("=" * 60)

            '''# === Evidently AI Data Drift Report ===
            # Remove target column if present (for drift, use features only)
            train_features = X_train.drop(columns=["Defaulter"], errors='ignore')
            test_features = X_test.copy()
            if "Defaulter" in test_features.columns:
                test_features = test_features.drop(columns=["Defaulter"])

            report = Report([DataDriftPreset()])
            report.run(reference_data=train_features, current_data=test_features)
            report_path = f"datadrift_{name.replace(' ', '_')}.html"
            #report.save_html(report_path)
            with open(report_path, "w") as f:
                f.write(report.as_html())
                mlflow.log_artifact(report_path, artifact_path="evidently_reports")
            # Log Evidently report as MLflow artifact
            mlflow.log_artifact(report_path, artifact_path="evidently_reports")
            
            new_df_features = new_df.copy()
            report.run(reference_data=train_features,current_data=new_df_features)
            report_path_new = f"datadrift_{name.replace(' ', '_')}_newdf.html"
            with open(report_path_new, "w") as f:
                f.write(report.as_html())
                mlflow.log_artifact(report_path, artifact_path="evidently_reports")'''

 
            if acc > best_score:
                best_score = acc
                best_overall_model = best_model
                best_overall_params = grid.best_params_
                best_run_id = run.info.run_id
                best_model_name = name
 
    # Register and Promote Best Model
   
    # Register and set alias
   
 
    if best_run_id:
        model_name = "BestLoanDefaultModel"
        model_uri = f"runs:/{best_run_id}/model"
 
        print(f"\n Registering best model '{best_model_name}' to Model Registry...")
        result = mlflow.register_model(model_uri=model_uri, name=model_name)
 
        client = MlflowClient()
 
    # Create model in registry if not exists
    try:
        client.get_registered_model(model_name)
    except:
        client.create_registered_model(model_name)
 
    # Register new version
    mv = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=best_run_id
    )
 
    # Transition the model to STAGING (or PRODUCTION)
    client.transition_model_version_stage(
        name=model_name,
        version=mv.version,
        stage="Staging",  # Or use "Production"
        archive_existing_versions=True
    )
 
    print(f"\n Model version {mv.version} for {model_name} is now in 'Staging'")
 
    # === Deploy: Load and use the model ===
    staged_model = mlflow.sklearn.load_model(f"models:/{model_name}/Staging")
    # Example inference
    preds = staged_model.predict(X_test)
    print("\n Inference on Test Data (First 10 Predictions):", preds[:10])
    return best_overall_model, best_overall_params
 