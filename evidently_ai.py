import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from sklearn.model_selection import train_test_split
 
#  Load dataset
df = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", sheet_name='Data')
 
#  Separate features and target
X = df.drop(['Personal Loan', 'ID'], axis=1)
y = df['Personal Loan']
 
#  Train-test split
train_df, test_df, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 
#  Load new/unseen data and align columns
new_df = pd.read_csv("C:/Users/Minfy.CHIRANJIBISILLA/Desktop/Risk Classification System/Building model/uploads/New Customer Bank_Personal_Loan.csv")
new_df = new_df[X.columns]  # Ensure same feature set
 
#  Report: Train vs Test
train_vs_test_report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset(),
    TargetDriftPreset()
])
train_vs_test_report.run(reference_data=train_df, current_data=test_df)
 
#  Report: old vs New
old_vs_new_report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset()
])
old_vs_new_report.run(reference_data=train_df, current_data=new_df)
 
#  Save reports
train_vs_test_report.save_html("report_train_vs_test.html")
old_vs_new_report.save_html("report_old_vs_new.html")
 
print("Reports saved: 'report_train_vs_test.html', 'report_old_vs_new.html'")
 
 
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently import ColumnMapping
 
# Load data
df = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx", sheet_name="Data")
new_df = pd.read_csv("C:/Users/Minfy.CHIRANJIBISILLA/Desktop/Risk Classification System/Building model/uploads/New Customer Bank_Personal_Loan.csv")
common_cols = list(set(df.columns).intersection(new_df.columns))
df_aligned = df[common_cols].copy()
new_df_aligned = new_df[common_cols].copy()
mapping = ColumnMapping(
    target=None,
    numerical_features=[col for col in common_cols if df[col].dtype in ["int64", "float64"]],
    categorical_features=[col for col in common_cols if df[col].dtype == "object"]
)
old_vs_new_report.run(reference_data=df_aligned, current_data=new_df_aligned, column_mapping=mapping)
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=df_aligned, current_data=new_df_aligned, column_mapping=mapping)
 
# Then extract drift scores from report.as_dict()
report_dict = report.as_dict()  # ← FIXED: this was report.as_dict() not report.as_dict (missing parentheses)
drift_result = None
for metric in report_dict.get("metrics", []):
    if "drift_by_columns" in metric.get("result", {}):
        drift_result = metric.get("result")
        break
if drift_result is None:
    raise ValueError(" Could not find 'drift_by_columns' in any metric result.")
drift_ratio = drift_result.get("share_of_drifted_columns", 0.0)
print(f"\n Overall Drift Score (Ratio): {drift_ratio:.2%}")
drift_data = []
for feature, values in drift_result["drift_by_columns"].items():
    score = values.get("statistic", {}).get("value", None)
    drifted = values.get("drift_detected", None)
    drift_data.append({
        "Feature": feature,
        "Drift Score": round(score, 4) if score is not None else None,
        "Drift Detected": drifted
    })
 
import json
print(json.dumps(report.as_dict(), indent=2))
 
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Drift")
 
mlflow.end_run()
with mlflow.start_run(run_name="drift_log_via_DataDriftTable"):
 
   
    mlflow.log_artifact("report_train_vs_test.html", artifact_path="evidently_reports")
    mlflow.log_artifact("report_old_vs_new.html", artifact_path="evidently_reports")
 
    # Log metrics
    drift_result = next(
        (m["result"] for m in report_dict["metrics"] if m.get("metric") == "DataDriftTable"),
        None
    )
    if drift_result:
        mlflow.log_metric("datadrift_overall_ratio", drift_result["share_of_drifted_columns"])
        for feature, vals in drift_result["drift_by_columns"].items():
            score = vals.get("drift_score")
            if score is not None:
                name = feature.replace(" ", "_").replace("(", "").replace(")", "")
                mlflow.log_metric(f"datadrift_{name}", round(score, 4))
 
 