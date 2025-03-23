import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# from ydata_profiling import ProfileReport
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset


def load_data(file_name, col_to_drop=None):
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "data")
    return pd.read_csv(os.path.join(data_path, file_name)).drop(columns=col_to_drop)


def create_model_basefile(input_df):
    model_basefile = load_data(
        input_df,
        col_to_drop=[
            "start_date",
            "strt_statn",
            "end_date",
            "end_statn",
            "start_lat",
            "start_lng",
            "end_lat",
            "end_lng",
            "start_date_formatted",
            "DATE",
        ],
    ).dropna()

    model_basefile["month"] = model_basefile["month"].astype("category")
    model_basefile["week_day"] = model_basefile["week_day"].astype("category")
    model_basefile["is_holiday"] = model_basefile["is_holiday"].astype("category")
    model_basefile["peak_time"] = model_basefile["peak_time"].astype("category")
    model_basefile.info()
    return model_basefile


def create_profile_report():
    model_basefile = create_model_basefile("df_to_preprocess.csv")
    model_basefile_sample = model_basefile.sample(frac=0.05, random_state=123)
    profile = ProfileReport(
        model_basefile_sample, title="Model Base File Sample Profiling Report"
    )
    profile.to_file("artifacts/model_basefile_sample_profile.html")


def create_drift_report():
    model_basefile = create_model_basefile("df_to_preprocess.csv")
    model_basefile_sample = model_basefile.sample(frac=0.05, random_state=123)
    X = model_basefile_sample.drop("duration", axis=1)
    y = model_basefile_sample["duration"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )
    X_train["duration"] = y_train
    X_test["duration"] = y_test
    drift_report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    drift_report.run(reference_data=X_train, current_data=X_test)
    drift_report.save_html("artifacts/data_drift_report.html")


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def x_test_to_csv():
    model_basefile = create_model_basefile("df_to_preprocess.csv")
    model_basefile_sample = model_basefile.sample(frac=0.05, random_state=123)
    X = model_basefile_sample.drop("duration", axis=1)
    y = model_basefile_sample["duration"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=123
    )
    X_test.to_csv("artifacts/X_test.csv", index=False)
    return X_test


def model_prediction(X_test, y_test):
    project_path = os.getcwd()
    artifacts_path = os.path.join(project_path, "artifacts")
    loaded_model = joblib.load(os.path.join(artifacts_path, "best_model.pkl"))

    y_pred = loaded_model.predict(X_test)
    print("Best Model Performance:")
    print(f"R2 Score: {r2_score(y_test, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.3f}")
    return pd.DataFrame({"duration": np.exp(y_test), "duration_pred": np.exp(y_pred)})

if __name__ == "__main__":
    # create_drift_report()
    x_test_to_csv()
