# -------------------------------------------------------------------------
# Predicting Road Collision Severity in GP: Streamlit Application 
# -------------------------------------------------------------------------

# -----------------------------------
# Stage 1: Imports and page setup
# -----------------------------------

# import Libraries
import streamlit as st
import pickle # loads the model
import pandas as pd # data manipulation
import plotly.graph_objects as go # creates the charts
import shap # explains the model predictions
import matplotlib.pyplot as plt # creates the charts
from pathlib import Path # builds file paths that work from any folder

# folder containing this script, so files load no matter where the app is launched from
BASE_DIR = Path(__file__).resolve().parent

# streamlit page configuration
st.set_page_config(page_title="UK Road Collision Severity Predictor", layout="centered") 

# set the page title
st.title("Road Collision Severity Predictor")

# -------------------------------------
# Stage 2: Loading the trained model
# -------------------------------------

# load the trained model bundle and build the SHAP explainer once, then reuse them across reruns
@st.cache_resource
def load_model():
    with open(BASE_DIR / "final_model.pkl", "rb") as file:
        save_model_data = pickle.load(file)
    explainer = shap.TreeExplainer(save_model_data["model"])
    return save_model_data, explainer

save_model_data, explainer = load_model()

model = save_model_data["model"]
threshold = save_model_data["threshold"]
feature_names = save_model_data["feature_names"]

# ADDED IN STAGE 8: Readable names for the model's raw column names, used on the SHAP waterfall chart
FEATURE_LABELS = {
    "road_type_1": "Road Type: Roundabout",
    "road_type_2": "Road Type: One Way Street",
    "road_type_3": "Road Type: Dual Carriageway",
    "road_type_6": "Road Type: Single Carriageway",
    "road_type_7": "Road Type: Slip Road",
    "road_type_9": "Road Type: Unknown",
    "urban_or_rural_area_1": "Area: Urban",
    "urban_or_rural_area_2": "Area: Rural",
    "light_conditions_1": "Light: Daylight",
    "light_conditions_4": "Light: Dark - Lights Lit",
    "light_conditions_5": "Light: Dark - Lights Unlit",
    "light_conditions_6": "Light: Dark - No Lighting",
    "light_conditions_7": "Light: Dark - Lighting Unknown",
    "weather_conditions_1": "Weather: Fine",
    "weather_conditions_2": "Weather: Raining",
    "weather_conditions_3": "Weather: Snowing",
    "weather_conditions_4": "Weather: Fine / High Winds",
    "weather_conditions_5": "Weather: Rain / High Winds",
    "weather_conditions_6": "Weather: Snow / High Winds",
    "weather_conditions_7": "Weather: Fog / Mist",
    "weather_conditions_8": "Weather: Other",
    "weather_conditions_9": "Weather: Unknown",
    "junction_control_-1": "Junction: Not at a junction / missing",
    "junction_control_1": "Junction: Traffic Signal",
    "junction_control_2": "Junction: Stop Sign",
    "junction_control_3": "Junction: Give Way / Uncontrolled",
    "junction_control_4": "Junction: Not Controlled",
    "junction_control_9": "Junction: Unknown",
    "speed_limit": "Speed Limit",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
}


# model suceesfully loaded: visual confirmation
st.write(f"Model loaded successfully! The threshold for classification is set at {threshold}.")

# --------------------------------
# Stage 3: Creating the tabs
# --------------------------------

# create the tabs for the application
tab_predict, tab_overview, tab_shap = st.tabs(["Risk Predictor", "Dataset Overview", "SHAP Explorer"])

with tab_predict:

    # ------------------------------------------------
    # Stage 4: The Risk Predictor's input widgets
    # ------------------------------------------------

    # set tab header
    st.header("Road & Conditions")

    # splits page into 2 columns
    col1, col2 = st.columns(2)

    
    with col1: # puts things in the left column
        speed_limit = st.selectbox("Speed Limit (mph)", options=[20, 30, 40, 50, 60, 70], index=1)
        road_type_label = st.selectbox(
            "Road Type",
            ["Roundabout", "One Way Street", "Dual Carriageway", "Single Carriageway", "Slip Road", "Unknown"],
            index=3
        )
        urban_rural_label = st.selectbox("Area Type", ["Urban", "Rural"])
        hour = st.slider("Hour of the Day", 0, 23, 17)

    with col2: # puts things in the right column
        light_label = st.selectbox(
            "Lighting Conditions",
            ["Daylight", "Dark - Lights Lit", "Dark - Lights Unlit", "Dark - No Lighting", "Dark - Lighting Unknown"],

        )
        weather_label = st.selectbox(
            "Weather",
            ["Fine", "Raining", "Snowing", "Fine / High Winds", "Rain / High Winds", "Snow / High Winds", "Fog / Mist", "Other", "Unknown"],
        )
        junction_control_label = st.selectbox(
            "Junction Control",
            ["Not at a junction / missing", "Traffic Signal", "Stop Sign", "Give Way / Uncontrolled",
            "Not Controlled", "Unknown"],
        )
        day_of_week_label = st.selectbox(
            "Day of Week", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        )
        month = st.slider("Month", 1, 12, 6) # a draggable slider for the month of the year

    st.divider() # draws a horizontal line to separate the sections

    # ------------------------------------------------
    # Stage 5: The predict button and the prediction output
    # ------------------------------------------------

    # create a button to trigger the prediction
    if st.button("Predict Severity Risk", type="primary"):

        # map the user input to the corresponding codes

        # road type
        if road_type_label == "Roundabout":
            road_type_code = 1
        elif road_type_label == "One Way Street":
            road_type_code = 2
        elif road_type_label == "Dual Carriageway":
            road_type_code = 3
        elif road_type_label == "Single Carriageway":
            road_type_code = 6
        elif road_type_label == "Slip Road":
            road_type_code = 7
        else:
            road_type_code = 9

        # urban/rural area
        if urban_rural_label == "Urban":
            urban_rural_code = 1
        else:
            urban_rural_code = 2

        # lighting conditions
        if light_label == "Daylight":
            light_code = 1
        elif light_label == "Dark - Lights Lit":
            light_code = 4
        elif light_label == "Dark - Lights Unlit":
            light_code = 5
        elif light_label == "Dark - No Lighting":
            light_code = 6
        else:
            light_code = 7

        # weather conditions
        if weather_label == "Fine":
            weather_code = 1
        elif weather_label == "Raining":
            weather_code = 2
        elif weather_label == "Snowing":
            weather_code = 3
        elif weather_label == "Fine / High Winds":
            weather_code = 4
        elif weather_label == "Rain / High Winds":
            weather_code = 5
        elif weather_label == "Snow / High Winds":
            weather_code = 6
        elif weather_label == "Fog / Mist":
            weather_code = 7
        elif weather_label == "Other":
            weather_code = 8
        else:
            weather_code = 9

        # junction control
        if junction_control_label == "Not at a junction / missing":
            junction_control_code = -1
        elif junction_control_label == "Traffic Signal":
            junction_control_code = 1
        elif junction_control_label == "Stop Sign":
            junction_control_code = 2
        elif junction_control_label == "Give Way / Uncontrolled":
            junction_control_code = 3
        elif junction_control_label == "Not Controlled":
            junction_control_code = 4
        else:
            junction_control_code = 9

        # day of week
        if day_of_week_label == "Sunday":
            day_of_week_code = 0
        elif day_of_week_label == "Monday":
            day_of_week_code = 1
        elif day_of_week_label == "Tuesday":
            day_of_week_code = 2
        elif day_of_week_label == "Wednesday":
            day_of_week_code = 3
        elif day_of_week_label == "Thursday":
            day_of_week_code = 4
        elif day_of_week_label == "Friday":
            day_of_week_code = 5
        else:
            day_of_week_code = 6

    # ------------------------------------------------
    # Stage 6: Building the model's input
    # ------------------------------------------------

        # create a DataFrame with the same structure as the training data
        input_row = pd.DataFrame(0, index=[0], columns=feature_names)

        # populate with the user input values
        input_row["speed_limit"] = speed_limit
        input_row["hour"] = hour
        input_row["day_of_week"] = day_of_week_code
        input_row["month"] = month

        # populate with one-hot encoded values for categorical features
        road_type_column = "road_type_" + str(road_type_code)
        if road_type_column in input_row.columns:
            input_row[road_type_column] = 1

        urban_rural_column = "urban_or_rural_area_" + str(urban_rural_code)
        if urban_rural_column in input_row.columns:
            input_row[urban_rural_column] = 1

        light_column = "light_conditions_" + str(light_code)
        if light_column in input_row.columns:
            input_row[light_column] = 1

        weather_column = "weather_conditions_" + str(weather_code)
        if weather_column in input_row.columns:
            input_row[weather_column] = 1

        junction_column = "junction_control_" + str(junction_control_code)
        if junction_column in input_row.columns:
            input_row[junction_column] = 1

    # ------------------------------------------------
    # Stage 7: Make the prediction and display the result
    # ------------------------------------------------
        
        # make the prediction using the trained model
        prediction = model.predict_proba(input_row)
        probability = prediction[0][1]
        is_high_risk = probability >= threshold

        # display the prediction result using a gauge chart
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%"},
                title={"text": "Probability of Serious/Fatal Outcome"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if is_high_risk else "darkgreen"},
                    "steps": [
                        {"range": [0, threshold * 100], "color": "#d4f4dd"},
                        {"range": [threshold * 100, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line": {"color": "black", "width": 3},
                        "thickness": 0.75,
                        "value": threshold * 100,
                    },
                },
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        # display the risk classification result
        if is_high_risk:
            st.error(f"HIGH RISK — above the {threshold:.1%} decision threshold")
        else:
            st.success(f"LOWER RISK — below the {threshold:.1%} decision threshold")

    # ------------------------------------------------
    # Stage 8: The SHAP waterfall chart 
    # ------------------------------------------------

    # READ: Readable labels dictionary added to 'explainer = shap.TreeExplainer(model)' in Stage 2

        # display the subheader for the chart
        st.subheader("Why This Prediction?")

        # calculate SHAP values for the input row
        shap_values = explainer(input_row)

        readable_names = [] # create a list to hold the readable names
        for column_name in input_row.columns: # iterate through the column names of the input row
            if column_name in FEATURE_LABELS: # check if the column name has a readable label
                readable_names.append(FEATURE_LABELS[column_name]) 
            else:
                readable_names.append(column_name)
        shap_values.feature_names = readable_names

        # display the SHAP waterfall chart using matplotlib
        fig2 = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(fig2)
        plt.clf()

with tab_overview:

    # ------------------------------------------------
    # Stage 9: Dataset Overview
    # ------------------------------------------------

     # ------------------------------------------------
    # Stage 9a: Summary statistics
    # ------------------------------------------------

    # load the dataset once and reuse it across reruns
    @st.cache_data
    def load_dataset():
        # slim copy of the full dataset containing only the needed columns
        columns_needed = ["collision_severity", "road_type", "speed_limit", "urban_or_rural_area", "collision_year"]

        # read the dataset into a DataFrame
        df = pd.read_csv(BASE_DIR.parent / "data" / "dataset_slim.csv", usecols=columns_needed)

        # Build a simple 0/1 target column: 1 = Serious or Fatal, 0 = Slight
        target_values = []
        for severity in df["collision_severity"]:
            if severity == 1 or severity == 2:
                target_values.append(1)
            else:
                target_values.append(0)
        df["target"] = target_values
        return df

    df = load_dataset()

    # create a header and information about the dataset
    st.header("About the Dataset")
    st.markdown(
        "Source: UK DfT STATS19 road collision records, Great Britain, 2020–2024. "
        "Severity is collapsed to a binary target — Serious/Fatal vs Slight."
    )

    # display key metrics about the dataset in three columns
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw collision records", f"{len(df):,}")
    m2.metric("Serious/Fatal rate", f"{df['target'].mean():.1%}")
    m3.metric("Date range", f"{df['collision_year'].min()}–{df['collision_year'].max()}")
    st.caption("503,022 records were used for model training/testing after cleaning (see notebook Sections 4–7).")

    # ------------------------------------------------
    # Stage 9b: Class imbalance chart
    # ------------------------------------------------

    st.subheader("Class Balance")
    counts = df["target"].value_counts()
    class_balance = pd.DataFrame({"Count": [counts[0], counts[1]]}, index=["Slight", "Serious/Fatal"])
    st.bar_chart(class_balance)
    st.caption("The ~77/23 imbalance is why the model uses class-weighting rather than the raw split — see Section 15.")

    # ------------------------------------------------
    # Stage 9c: Severity rate by top predictors
    # ------------------------------------------------

    st.subheader("Severity Rate by Top Predictors")
    st.caption("These three ranked highest in the SHAP feature importance — this is the pattern the model learned.")

    c1, c2, c3 = st.columns(3)

    # display the bar charts for each of the top predictors in their respective columns

    with c1:
        # display the bar chart for road type
        st.write("**Road Type**")
        road_type_names = {1: "Roundabout", 2: "One Way Street", 3: "Dual Carriageway",
                            6: "Single Carriageway", 7: "Slip Road", 9: "Unknown"}
        rate_by_road = df.groupby("road_type")["target"].mean().sort_values(ascending=False)
        new_index = []
        for code in rate_by_road.index:
            new_index.append(road_type_names[code])
        rate_by_road.index = new_index
        st.bar_chart(rate_by_road)

    with c2:
        # display the bar chart for speed limit
        st.write("**Speed Limit**")
        rate_by_speed = df.groupby("speed_limit")["target"].mean()
        st.bar_chart(rate_by_speed)

    with c3:
        # display the bar chart for urban vs rural
        st.write("**Urban vs Rural**")
        urban_rural_df = df[(df["urban_or_rural_area"] == 1) | (df["urban_or_rural_area"] == 2)]
        rate_by_area = urban_rural_df.groupby("urban_or_rural_area")["target"].mean()
        new_index = []
        for code in rate_by_area.index:
            if code == 1:
                new_index.append("Urban")
            else:
                new_index.append("Rural")
        rate_by_area.index = new_index
        st.bar_chart(rate_by_area)

with tab_shap:

    # ------------------------------------------------
    # Stage 10: SHAP Explorer
    # ------------------------------------------------

    st.header("What Drives the Model's Predictions")
    st.markdown(
        "These charts summarise the model's behaviour across the full test set, "
        "rather than a single prediction — computed via SHAP (SHapley Additive "
        "exPlanations) on the final class-weighted XGBoost model."
    )

    # display the SHAP feature importance chart
    st.subheader("Feature Importance")
    st.image(str(BASE_DIR.parent / "graphs" / "shap_feature_importance.png"))
    st.caption(
        "Average magnitude of each feature's effect on the model's output, "
        "across a 5,000-row sample of the test set. Road Type, Speed Limit, and "
        "Rural Area dominate — see the Dataset Overview tab for why."
    )

    # display the SHAP dot plot chart
    st.subheader("Direction of Effect")
    st.image(str(BASE_DIR.parent / "graphs" / "shap_dot_plot.png"))
    st.caption(
        "Each dot is one collision. Red = high value for that feature, blue = low. "
        "Position right of centre pushes the prediction toward Serious/Fatal; left "
        "pushes toward Slight. E.g. high Speed Limit (red, right) increases risk; "
        "Junction Control: Stop Sign (red, left) is protective."
    )


# ----------------------------------------------------------------------------------






