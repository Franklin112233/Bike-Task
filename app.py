import streamlit as st
import pandas as pd

import numpy as np
from PIL import Image
from pathlib import Path
import streamlit.components.v1 as components
import json
import os
import joblib

# @st.cache_data
# def read_test_csv():
#     project_path = os.getcwd()
#     data_path = os.path.join(project_path, "artifacts")
#     return pd.read_csv(os.path.join(data_path, "X_test.csv"))


# @st.cache_data
# def read_station_csv():
#     project_path = os.getcwd()
#     data_path = os.path.join(project_path, "artifacts")
#     return pd.read_csv(os.path.join(data_path, "hubway_stations.csv"))


@st.cache_data
def load_data_app(file_name):
    project_path = os.getcwd()
    data_path = os.path.join(project_path, "artifacts")
    return pd.read_csv(os.path.join(data_path, file_name))


def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    return c * r


def web_app(st):
    st.set_page_config(
        page_title="RAPP BikeHub Task Demo",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )

    html_temp = """
    <div style="background: #1868a6;padding:10px">
    <h1 style="color:white;text-align:center;">BikeHub Task Demo</h1>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    logo = Image.open("artifacts/logo.png")
    st.sidebar.image(logo)

    if st.sidebar.button("About"):
        st.sidebar.text(
            """
        RAPP BikeHub Task Demo V1.0
        by Linlin Yu
        Mar 2025
        """
        )

    page = st.sidebar.selectbox(
        "Navigation", ["Overview", "Data Exploration", "ML Model", "Prediction"]
    )

    if page == "Overview":
        st.header("Data Overview")
        st.write("Upload or view the bike sharing dataset")

    elif page == "Data Exploration":
        st.header("Data Exploration")
        plot_type = st.radio(
            "Select",
            [
                "Data Profile Report",
                "Duration and Time",
                "Distance between Stations",
                "Subscription Gender Age",
            ],
        )

        if plot_type == "Data Profile Report":
            with Path("artifacts/model_basefile_sample_profile.html").open() as f:
                html_data = f.read()
            components.html(html_data, scrolling=True, height=700)

        if plot_type == "Duration and Time":
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)
            duration_boxplot = Image.open("artifacts/duration_boxplot.png")
            duration_by_day = Image.open("artifacts/duration_by_day_of_week.png")
            duration_by_holiday = Image.open("artifacts/duration_by_holiday.png")
            duration_by_peak = Image.open("artifacts/duration_by_peak_time.png")
            with col1:
                st.image(duration_boxplot, caption="Distribution of Trip Duration")
            with col2:
                st.image(duration_by_day, caption="Trip Duration by Day of Week")
            with col3:
                st.image(
                    duration_by_holiday,
                    caption="Trip Duration on Holidays vs Non-Holidays",
                )
            with col4:
                st.image(duration_by_peak, caption="Trip Duration by Peak Hours")

        if plot_type == "Distance between Stations":
            col1, col2 = st.columns(2)
            distance_distribution = Image.open("artifacts/distance_distribution.png")
            duration_by_distance = Image.open("artifacts/duration_by_distance.png")
            with col1:
                st.image(distance_distribution, caption="Distance Distribution")
            with col2:
                st.image(duration_by_distance, caption="Duration by Distance")

            station_df = load_data_app("hubway_stations.csv")
            unique_value = station_df["station"].unique()
            station_1 = st.selectbox("Select Start Station", unique_value)
            station_2 = st.selectbox("Select End Station", unique_value)
            if station_1 and station_2:
                station_1_lat = station_df[station_df["station"] == station_1][
                    "lat"
                ].values[0]
                station_1_lng = station_df[station_df["station"] == station_1][
                    "lng"
                ].values[0]
                station_2_lat = station_df[station_df["station"] == station_2][
                    "lat"
                ].values[0]
                station_2_lng = station_df[station_df["station"] == station_2][
                    "lng"
                ].values[0]
                distance = round(
                    calculate_distance(
                        station_1_lat, station_1_lng, station_2_lat, station_2_lng
                    ),
                    3,
                )
                st.success(
                    f"The distance between {station_1} and {station_2} is {distance} km"
                )

        if plot_type == "Subscription Gender Age":
            col1, col2 = st.columns(2)
            subscription_gender = Image.open(
                "artifacts/duration_by_subscription_gender.png"
            )
            duration_by_age = Image.open("artifacts/duration_by_age.png")
            with col1:
                st.image(subscription_gender, caption="Subscription Gender Age")
            with col2:
                st.image(duration_by_age, caption="Duration by Age Group")

    elif page == "ML Model":
        st.header("Machine Learning Model")
        plot_type = st.radio(
            "Select",
            [
                "Data Drift Detection",
                "Model Techniques and Metrics",
            ],
        )

        if plot_type == "Data Drift Detection":
            with Path("artifacts/data_drift_report.html").open() as f:
                html_data = f.read()
            components.html(html_data, scrolling=True, height=700)

        if plot_type == "Model Techniques and Metrics":
            st.markdown(
                """
                - Sample the data to reduce the training time (5% of the data)
                - Split the data into training and test sets (75% for training, 25% for testing)
                - Define a preprocessor for the model (standardize the numerical features, one-hot encode the categorical features)
                - Define a model pipeline including the preprocessor and the regressor
                - Three types of regressors included: Linear Regression, Random Forest, and XGBoost
                - Define the search spaces for each regressor for the hyperparameters tuning
                - Define the bayes search with cross validation for the model
                - Model performance metrics: R-squared and RMSE
                """
            )

            with open("artifacts/all_model_scores.json", "r") as f:
                all_model_scores = json.load(f)
            col1, col2 = st.columns(2)
            with col1:
                st.json(all_model_scores)
            with col2:
                feature_importance = Image.open("artifacts/feature_importance.png")
                st.image(feature_importance, caption="Feature Importance")

    else:
        st.header("Make Predictions")
        x_test = load_data_app("X_test.csv")
        col1, col2 = st.columns(2)

        with col1:
            start_municipal = st.selectbox(
                "Start Municipal", options=sorted(x_test["start_municipal"].unique())
            )

            start_status = st.selectbox(
                "Start Status", options=sorted(x_test["start_status"].unique())
            )

            end_municipal = st.selectbox(
                "End Municipal", options=sorted(x_test["end_municipal"].unique())
            )

            end_status = st.selectbox(
                "End Status", options=sorted(x_test["end_status"].unique())
            )

            HPCP = st.selectbox("HPCP", options=sorted(x_test["HPCP"].unique()))

            month = st.selectbox("Month", options=sorted(x_test["month"].unique()))

            week_day = st.selectbox(
                "Week Day", options=sorted(x_test["week_day"].unique())
            )

        with col2:
            is_holiday = st.selectbox(
                "Is Holiday", options=sorted(x_test["is_holiday"].unique())
            )

            peak_time = st.selectbox(
                "Peak Time", options=sorted(x_test["peak_time"].unique())
            )

            stations_df = load_data_app("hubway_stations.csv")

            start_station = st.selectbox(
                "Start Station", options=sorted(stations_df["station"].unique())
            )

            end_station = st.selectbox(
                "End Station", options=sorted(stations_df["station"].unique())
            )

            if start_station and end_station:
                station_df = load_data_app("hubway_stations.csv")
                start_station_lat = stations_df[
                    stations_df["station"] == start_station
                ]["lat"].values[0]
                start_station_lng = station_df[station_df["station"] == start_station][
                    "lng"
                ].values[0]
                end_station_lat = station_df[station_df["station"] == end_station][
                    "lat"
                ].values[0]
                end_station_lng = station_df[station_df["station"] == end_station][
                    "lng"
                ].values[0]
                distance = round(
                    calculate_distance(
                        start_station_lat,
                        start_station_lng,
                        end_station_lat,
                        end_station_lng,
                    ),
                    3,
                )

            subscription_gender = st.selectbox(
                "Subscription Gender",
                options=sorted(x_test["subscription_gender"].unique()),
            )

            age = st.selectbox("Age", options=sorted(x_test["age"].unique()))

            test_case = pd.DataFrame(
                {
                    "start_municipal": [start_municipal],
                    "start_status": [start_status],
                    "end_municipal": [end_municipal],
                    "end_status": [end_status],
                    "HPCP": [HPCP],
                    "month": [month],
                    "week_day": [week_day],
                    "is_holiday": [is_holiday],
                    "peak_time": [peak_time],
                    "distance": [distance],
                    "subscription_gender": [subscription_gender],
                    "age": [age],
                }
            )

            project_path = os.getcwd()
            artifacts_path = os.path.join(project_path, "artifacts")
            loaded_model = joblib.load(os.path.join(artifacts_path, "best_model.pkl"))

            predict_button = st.button("Predict Duration")
            if predict_button:
                y_pred = np.exp(loaded_model.predict(test_case))
            else:
                st.stop()
            st.success(f"Predicted Duration: {y_pred[0]:.2f} seconds")


if __name__ == "__main__":
    web_app(st)
