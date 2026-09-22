# -------------------------------------------------------------------------
# Predicting Road Collision Severity in GP: Streamlit Application
# -------------------------------------------------------------------------

# -----------------------------------
# Stage 1: Imports and page setup
# -----------------------------------

# import Libraries
import streamlit as st
import pickle # loads the model
import numpy as np # numerical helpers for the charts
import pandas as pd # data manipulation
import plotly.graph_objects as go # creates the charts
import shap # explains the model predictions
from pathlib import Path # builds file paths that work from any folder

# folder containing this script, so files load no matter where the app is launched from
BASE_DIR = Path(__file__).resolve().parent

# streamlit page configuration
st.set_page_config(page_title="UK Road Collision Severity Predictor", layout="wide")

# ADDED IN REDESIGN: styling that the theme in .streamlit/config.toml can't reach on its own
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', sans-serif; }
h1, h2, h3, h4 { font-weight: 500 !important; letter-spacing: -0.015em; }
h1 { font-size: 42px !important; }
h2 { font-size: 32px !important; }
h4 { font-size: 20px !important; }
.block-container { padding: 2.5rem 3.5rem 3rem; max-width: 1280px; }
/* outlined primary button */
.stButton > button[kind="primary"] { background: transparent; color: #9184d9; border: 1px solid #9184d9; width: 100%; }
.stButton > button[kind="primary"] p { font-size: 14px; font-weight: 500; }
.stButton > button[kind="primary"]:hover { background: rgba(145,132,217,.12); color: #9184d9; border-color: #9184d9; }
.stButton > button[kind="primary"]:active { background: rgba(145,132,217,.22); color: #9184d9; }
/* tabs */
.stTabs [data-baseweb="tab-list"] { gap: 28px; margin-top: 12px; }
.stTabs [data-baseweb="tab"] { color: #9397ab; font-size: 14px; }
.stTabs [data-baseweb="tab"] p { font-size: 14px; }
.stTabs [aria-selected="true"] { color: #9184d9; }
.stTabs [data-baseweb="tab-highlight"] { background: #9184d9; }
.stTabs [data-baseweb="tab-border"] { background: rgba(233,233,237,.16); height: 1px; }
/* inputs card */
.st-key-inputs_card { background: #232532; padding: 22px; }
[data-testid="stWidgetLabel"] p { font-size: 12px; color: rgba(233,233,237,.7); }
[data-baseweb="select"] > div { background: #161826; border: 1px solid rgba(233,233,237,.16); min-height: 36px; }
[data-baseweb="select"] div { font-size: 14px; }
[data-baseweb="select"] > div:hover { border-color: rgba(233,233,237,.45); }
[data-baseweb="select"] > div:focus-within { border-color: #9184d9; }
[data-testid="stSliderThumbValue"] { color: #d2cefd; font-size: 12px; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #9397ab; font-size: 11px; }
/* dataset overview cards */
[data-testid="stMetric"] { background: #232532; }
[data-testid="stMetric"] label p { color: #9397ab; font-size: 13px; }
[data-testid="stMetricValue"] { font-size: 36px; font-weight: 500; }
.st-key-metric_rate [data-testid="stMetricValue"] { color: #d2cefd; }
[class*="st-key-chart_card"] { background: #232532; }
[data-testid="stCaptionContainer"] { font-size: 12px; }
</style>""", unsafe_allow_html=True)

# ADDED IN REDESIGN: shared colours and chart styling so every Plotly chart matches the theme
NOC = dict(bg="rgba(0,0,0,0)", text="#cfd3e5", muted="#75798c", grid="#3f424d",
           accent="#9184d9", neg="#75798c", hi="#d2cefd")

def noc_layout(fig, h):
    fig.update_layout(height=h, paper_bgcolor=NOC["bg"], plot_bgcolor=NOC["bg"],
        font=dict(family="Inter, sans-serif", size=12, color=NOC["text"]),
        margin=dict(l=0, r=10, t=10, b=30), showlegend=False, bargap=0.35)
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)", zerolinecolor=NOC["grid"], tickfont_color=NOC["muted"])
    fig.update_yaxes(gridcolor="rgba(0,0,0,0)", tickfont_color=NOC["text"])
    return fig

# chart settings used for every st.plotly_chart call
CHART_CONFIG = {"displayModeBar": False}

# opacity for each bar, scaled so the largest value is the brightest
def scaled_opacity(values, low, high):
    values = np.asarray(values, dtype=float)
    spread = values.max() - values.min()
    if spread == 0:
        return [high] * len(values)
    return list(low + (values - values.min()) / spread * (high - low))

# Phosphor icons (regular weight) used in the page header and the empty state
ICON_CHECK_CIRCLE = ('<svg width="15" height="15" viewBox="0 0 256 256" fill="#9184d9" style="flex:none">'
    '<path d="M173.66,98.34a8,8,0,0,1,0,11.32l-56,56a8,8,0,0,1-11.32,0l-24-24a8,8,0,0,1,11.32-11.32L112,148.69l50.34-50.35A8,8,0,0,1,173.66,98.34ZM232,128A104,104,0,1,1,128,24,104.11,104.11,0,0,1,232,128Zm-16,0a88,88,0,1,0-88,88A88.1,88.1,0,0,0,216,128Z"></path></svg>')
ICON_CHART_BAR = ('<svg width="28" height="28" viewBox="0 0 256 256" fill="#75798c">'
    '<path d="M224,200h-8V40a8,8,0,0,0-8-8H152a8,8,0,0,0-8,8V80H96a8,8,0,0,0-8,8v40H48a8,8,0,0,0-8,8v64H32a8,8,0,0,0,0,16H224a8,8,0,0,0,0-16ZM160,48h40V200H160ZM104,96h40V200H104ZM56,144H88v56H56Z"></path></svg>')

# a 1px rule that fades out over 48px at each end
FADING_RULE = ('<div style="height:1px;margin:28px 0 22px;background:linear-gradient(to right, transparent, '
    'rgba(233,233,237,.16) 48px, rgba(233,233,237,.16) calc(100% - 48px), transparent)"></div>')

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

# ADDED IN STAGE 8: Readable names for the model's raw column names, used on the SHAP charts
# (same mapping as the notebook, covering all 83 model features)
FEATURE_LABELS = {
    # first road class (reference: Motorway)
    "first_road_class_2": "First Road: A(M)",
    "first_road_class_3": "First Road: A Road",
    "first_road_class_4": "First Road: B Road",
    "first_road_class_5": "First Road: C Road",
    "first_road_class_6": "First Road: Unclassified",
    # second road class (reference: -1 / not applicable)
    "second_road_class_0": "Second Road: No Junction",
    "second_road_class_1": "Second Road: Motorway",
    "second_road_class_2": "Second Road: A(M)",
    "second_road_class_3": "Second Road: A Road",
    "second_road_class_4": "Second Road: B Road",
    "second_road_class_5": "Second Road: C Road",
    "second_road_class_6": "Second Road: Unclassified",
    # road type (reference: Roundabout)
    "road_type_2": "Road Type: One Way Street",
    "road_type_3": "Road Type: Dual Carriageway",
    "road_type_6": "Road Type: Single Carriageway",
    "road_type_7": "Road Type: Slip Road",
    "road_type_9": "Road Type: Unknown",
    # junction detail (reference: Not at junction)
    "junction_detail_13": "Junction: T or Staggered",
    "junction_detail_16": "Junction: Crossroads",
    "junction_detail_17": "Junction: More Than 4 Arms",
    "junction_detail_18": "Junction: Private Drive",
    "junction_detail_19": "Junction: Other Junction",
    "junction_detail_99": "Junction: Unknown",
    # junction control (reference: -1 / data missing)
    "junction_control_1": "Junction Control: Traffic Signal",
    "junction_control_2": "Junction Control: Stop Sign",
    "junction_control_3": "Junction Control: Give Way / Uncontrolled",
    "junction_control_4": "Junction Control: Not Controlled",
    "junction_control_9": "Junction Control: Unknown",
    # pedestrian crossing (reference: Zebra / pelican)
    "pedestrian_crossing_0": "Pedestrian Crossing: None",
    "pedestrian_crossing_11": "Pedestrian Crossing: Pelican / Puffin / Toucan",
    "pedestrian_crossing_12": "Pedestrian Crossing: Traffic Signal Phase",
    "pedestrian_crossing_13": "Pedestrian Crossing: Footbridge / Subway",
    "pedestrian_crossing_14": "Pedestrian Crossing: Central Refuge",
    "pedestrian_crossing_15": "Pedestrian Crossing: School Patrol",
    "pedestrian_crossing_16": "Pedestrian Crossing: Zig-zag Markings",
    "pedestrian_crossing_17": "Pedestrian Crossing: Other Controlled",
    "pedestrian_crossing_99": "Pedestrian Crossing: Unknown",
    # light conditions (reference: Daylight)
    "light_conditions_4": "Light: Dark - Lights Lit",
    "light_conditions_5": "Light: Dark - Lights Unlit",
    "light_conditions_6": "Light: Dark - No Lighting",
    "light_conditions_7": "Light: Dark - Lighting Unknown",
    # weather conditions (reference: Fine)
    "weather_conditions_2": "Weather: Raining",
    "weather_conditions_3": "Weather: Snowing",
    "weather_conditions_4": "Weather: Fine / High Winds",
    "weather_conditions_5": "Weather: Rain / High Winds",
    "weather_conditions_6": "Weather: Snow / High Winds",
    "weather_conditions_7": "Weather: Fog / Mist",
    "weather_conditions_8": "Weather: Other",
    "weather_conditions_9": "Weather: Unknown",
    # road surface conditions (reference: Dry)
    "road_surface_conditions_2": "Road Surface: Wet / Damp",
    "road_surface_conditions_3": "Road Surface: Snow",
    "road_surface_conditions_4": "Road Surface: Frost / Ice",
    "road_surface_conditions_5": "Road Surface: Flood",
    "road_surface_conditions_9": "Road Surface: Unknown",
    # special conditions (reference: -1 / data missing)
    "special_conditions_at_site_0": "Special Conditions: None",
    "special_conditions_at_site_1": "Special Conditions: Auto Signal Out",
    "special_conditions_at_site_2": "Special Conditions: Auto Signal Defective",
    "special_conditions_at_site_3": "Special Conditions: Roadworks",
    "special_conditions_at_site_4": "Special Conditions: Road Surface Defective",
    "special_conditions_at_site_5": "Special Conditions: Oil / Diesel",
    "special_conditions_at_site_6": "Special Conditions: Mud",
    "special_conditions_at_site_7": "Special Conditions: Other",
    "special_conditions_at_site_9": "Special Conditions: Unknown",
    # carriageway hazards (reference: -1 / data missing)
    "carriageway_hazards_0": "Carriageway Hazard: None",
    "carriageway_hazards_11": "Carriageway Hazard: Vehicle Load on Road",
    "carriageway_hazards_12": "Carriageway Hazard: Other Object on Road",
    "carriageway_hazards_13": "Carriageway Hazard: Previous Accident",
    "carriageway_hazards_14": "Carriageway Hazard: Dog on Road",
    "carriageway_hazards_15": "Carriageway Hazard: Other Animal",
    "carriageway_hazards_16": "Carriageway Hazard: Pedestrian in Road",
    "carriageway_hazards_17": "Carriageway Hazard: Fallen Cyclist",
    "carriageway_hazards_18": "Carriageway Hazard: Other",
    "carriageway_hazards_19": "Carriageway Hazard: Unknown",
    "carriageway_hazards_20": "Carriageway Hazard: Not Reported",
    "carriageway_hazards_21": "Carriageway Hazard: Reported Missing",
    "carriageway_hazards_99": "Carriageway Hazard: Not Recorded",
    # urban/rural (reference: Urban)
    "urban_or_rural_area_2": "Rural Area",
    # trunk road (reference: -1 / data missing)
    "trunk_road_flag_1": "Non-Trunk Road",
    "trunk_road_flag_2": "Trunk Road",
    # numeric features
    "speed_limit": "Speed Limit",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
}


# model suceesfully loaded: visual confirmation
st.markdown(
    '<div style="display:flex;align-items:center;gap:6px;font-size:13px;color:#9397ab;margin-top:-8px">'
    f'{ICON_CHECK_CIRCLE}<span>Model loaded successfully! The threshold for classification is set at {threshold}.</span></div>',
    unsafe_allow_html=True,
)

# --------------------------------
# Stage 3: Creating the tabs
# --------------------------------

# create the tabs for the application
tab_predict, tab_overview, tab_shap = st.tabs(["Risk Predictor", "Dataset Overview", "SHAP Explorer"])

with tab_predict:

    # ------------------------------------------------
    # Stage 4: The Risk Predictor's input widgets
    # ------------------------------------------------

    # splits page into the inputs card (left) and the result (right)
    left, right = st.columns([0.38, 0.62], gap="medium")

    with left:
        with st.container(border=True, key="inputs_card"):

            # set card header
            st.markdown("#### Road & Conditions")

            # two fields per row
            a, b = st.columns(2)
            with a:
                speed_limit = st.selectbox("Speed Limit (mph)", options=[20, 30, 40, 50, 60, 70], index=1)
            with b:
                road_type_label = st.selectbox(
                    "Road Type",
                    ["Roundabout", "One Way Street", "Dual Carriageway", "Single Carriageway", "Slip Road", "Unknown"],
                    index=3
                )

            a, b = st.columns(2)
            with a:
                urban_rural_label = st.selectbox("Area Type", ["Urban", "Rural"])
            with b:
                light_label = st.selectbox(
                    "Lighting Conditions",
                    ["Daylight", "Dark - Lights Lit", "Dark - Lights Unlit", "Dark - No Lighting", "Dark - Lighting Unknown"],
                )

            a, b = st.columns(2)
            with a:
                weather_label = st.selectbox(
                    "Weather",
                    ["Fine", "Raining", "Snowing", "Fine / High Winds", "Rain / High Winds", "Snow / High Winds", "Fog / Mist", "Other", "Unknown"],
                )
            with b:
                junction_control_label = st.selectbox(
                    "Junction Control",
                    ["Not at a junction / missing", "Traffic Signal", "Stop Sign", "Give Way / Uncontrolled",
                    "Not Controlled", "Unknown"],
                )

            # full-width fields
            day_of_week_label = st.selectbox(
                "Day of Week", ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            )
            hour = st.slider("Hour of the Day", 0, 23, 17)
            month = st.slider("Month", 1, 12, 6) # a draggable slider for the month of the year

            # ------------------------------------------------
            # Stage 5: The predict button and the prediction output
            # ------------------------------------------------

            # create a button to trigger the prediction
            predict_clicked = st.button("Predict Severity Risk", type="primary", use_container_width=True)

    if predict_clicked:

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
    # Stage 7: Make the prediction and keep the result
    # ------------------------------------------------

        # make the prediction using the trained model
        prediction = model.predict_proba(input_row)
        probability = prediction[0][1]

        # calculate SHAP values for the input row
        shap_values = explainer(input_row)

        # keep the result in session state so it survives reruns and the right column can show an empty state
        st.session_state["result"] = {"probability": probability, "shap": shap_values[0]}

    # ------------------------------------------------
    # Stage 8: The result and the SHAP waterfall chart
    # ------------------------------------------------

    # READ: Readable labels dictionary added to 'explainer = shap.TreeExplainer(model)' in Stage 2

    # big percentage and risk tag for a prediction
    def result_html(p, t):
        hi = p >= t
        d = abs(p - t) * 100
        num_style = "color:#d2cefd;text-shadow:0 0 40px rgba(145,132,217,.35)" if hi else "color:#e4e7f5"
        tag = ('<span style="border:1px solid #9184d9;color:#9184d9;padding:4px 12px;border-radius:6px;font-size:12px;letter-spacing:.08em">HIGH RISK</span>' if hi
               else '<span style="background:#3f424d;color:#f3f5fe;padding:4px 12px;border-radius:6px;font-size:12px;letter-spacing:.08em">LOWER RISK</span>')
        word = "above" if hi else "below"
        return (
            '<div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#9184d9">Probability of Serious/Fatal Outcome</div>'
            f'<div style="font-size:112px;line-height:.95;font-weight:500;letter-spacing:-.045em;margin:14px 0 18px;{num_style}">{p*100:.1f}%</div>'
            f'<div style="display:flex;gap:10px;align-items:center">{tag}<span style="font-size:14px;color:#cfd3e5">{d:.1f} points {word} the {t:.1%} decision threshold</span></div>'
        )

    # the same block before Predict has been pressed
    def empty_result_html(t):
        return (
            '<div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#9184d9">Probability of Serious/Fatal Outcome</div>'
            '<div style="font-size:112px;line-height:.95;font-weight:500;letter-spacing:-.045em;margin:14px 0 18px;color:#3f424d">&mdash;%</div>'
            '<div style="display:flex;gap:10px;align-items:center">'
            '<span style="border:1px dashed #595d6c;color:#9397ab;padding:4px 12px;border-radius:6px;font-size:12px;letter-spacing:.08em">NO PREDICTION YET</span>'
            f'<span style="font-size:14px;color:#9397ab">Decision threshold: {t:.1%}</span></div>'
            + FADING_RULE +
            '<div style="border:1px dashed #3f424d;border-radius:8px;min-height:260px;padding:28px 32px;display:flex;flex-direction:column;justify-content:center;gap:8px">'
            f'<div>{ICON_CHART_BAR}</div>'
            '<div style="font-size:20px;font-weight:500;color:#cfd3e5">Why This Prediction?</div>'
            '<div style="font-size:13px;color:#9397ab;max-width:420px">Set the road and conditions on the left, then press Predict Severity Risk. '
            "The probability and a breakdown of each feature's contribution will appear here.</div></div>"
        )

    # waterfall chart of one prediction's SHAP values (replaces shap.plots.waterfall)
    def waterfall_fig(explanation, top_n=11):
        values = explanation.values
        base = float(explanation.base_values)
        order = np.argsort(-np.abs(values)) # largest contribution first
        top = order[:top_n]
        rest = order[top_n:]

        # rows are built bottom to top, so the running total starts at the base value
        labels, deltas = [], []
        if len(rest) > 0:
            labels.append(f"{len(rest)} other features")
            deltas.append(values[rest].sum())
        for i in top[::-1]:
            name = FEATURE_LABELS.get(feature_names[i], feature_names[i])
            labels.append(f"{explanation.data[i]:g} = {name}")
            deltas.append(values[i])

        starts = base + np.concatenate([[0], np.cumsum(deltas)[:-1]])
        colours = [NOC["accent"] if d >= 0 else NOC["neg"] for d in deltas]
        rows = list(range(len(labels)))

        fig = go.Figure(go.Bar(
            orientation="h", y=rows, x=deltas, base=starts,
            marker=dict(color=colours, cornerradius=2),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[f"{d:+.3f}" for d in deltas],
        ))
        fig.add_vline(x=base, line_color=NOC["grid"], line_width=1)

        # contribution values in a column on the right, e.g. +0.31 / −0.12
        for row, d in zip(rows, deltas):
            fig.add_annotation(x=1, xref="paper", y=row, xanchor="left", xshift=12, showarrow=False,
                text=f"{d:+.2f}".replace("-", "−"),
                font=dict(size=12, color=NOC["hi"] if d >= 0 else NOC["text"]))

        noc_layout(fig, 25 * len(labels) + 30)
        fig.update_layout(margin=dict(l=0, r=56, t=5, b=5), bargap=0.45)
        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(tickvals=rows, ticktext=labels, ticksuffix="  ")
        return fig

    with right:
        result = st.session_state.get("result")

        if result is None:
            st.markdown(empty_result_html(threshold), unsafe_allow_html=True)
        else:
            explanation = result["shap"]
            base = float(explanation.base_values)
            fx = base + explanation.values.sum()

            st.markdown(result_html(result["probability"], threshold) + FADING_RULE, unsafe_allow_html=True)

            # display the subheader for the chart
            st.markdown("#### Why This Prediction?")
            st.markdown(
                f'<div style="font-size:13px;color:#9397ab;margin-top:-8px">Feature contributions, log-odds, '
                f'E[f(x)] = {round(base, 2) + 0.0:.2f} → f(x) = {round(fx, 2) + 0.0:.2f}</div>'.replace("= -", "= −"),
                unsafe_allow_html=True,
            )

            # display the SHAP waterfall chart
            st.plotly_chart(waterfall_fig(explanation), use_container_width=True, config=CHART_CONFIG)

# ------------------------------------------------
# Stage 9: Dataset Overview
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

# bar chart of a severity rate, brighter bars for higher rates
def rate_bar_fig(rates, horizontal, h):
    labels = [str(label) for label in rates.index]
    opacity = scaled_opacity(rates.values, 0.45, 1.0)
    text = [f"{v:.1%}" for v in rates.values]
    if horizontal:
        bar = go.Bar(orientation="h", y=labels, x=rates.values, text=text)
    else:
        bar = go.Bar(x=labels, y=rates.values, text=text)
    bar.update(marker=dict(color=NOC["accent"], opacity=opacity, cornerradius=3),
               textposition="outside", textfont=dict(size=10.5, color="#9397ab"), constraintext="none",
               cliponaxis=False, hovertemplate="%{text}<extra></extra>")
    fig = noc_layout(go.Figure(bar), h)
    fig.update_layout(margin=dict(l=0 if horizontal else 4, r=40 if horizontal else 4, t=16 if horizontal else 20, b=0))
    if horizontal:
        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(autorange="reversed", ticksuffix="  ", tickfont_size=12)
    else:
        fig.update_yaxes(showticklabels=False, zeroline=False)
        fig.update_xaxes(type="category", tickfont_color=NOC["text"])
    return fig

with tab_overview:

    # ------------------------------------------------
    # Stage 9a: Summary statistics
    # ------------------------------------------------

    df = load_dataset()

    # create a header and information about the dataset
    st.header("About the Dataset")
    st.markdown(
        '<div style="font-size:14px;color:#cfd3e5;max-width:720px">'
        "Source: UK DfT STATS19 road collision records, Great Britain, 2020–2024. "
        "Severity is collapsed to a binary target — Serious/Fatal vs Slight.</div>",
        unsafe_allow_html=True,
    )

    # display key metrics about the dataset in three columns
    m1, m2, m3 = st.columns(3)
    m1.metric("Raw collision records", f"{len(df):,}", border=True)
    with m2.container(key="metric_rate"):
        st.metric("Serious/Fatal rate", f"{df['target'].mean():.1%}", border=True)
    m3.metric("Date range", f"{df['collision_year'].min()}–{df['collision_year'].max()}", border=True)
    st.caption("503,022 records were used for model training/testing after cleaning (see notebook Sections 4–7).")

    balance_col, predictors_col = st.columns([1, 2], gap="large")

    # ------------------------------------------------
    # Stage 9b: Class imbalance chart
    # ------------------------------------------------

    with balance_col:
        st.markdown("#### Class Balance")
        counts = df["target"].value_counts()
        total = counts.sum()

        # one labelled bar per class, width relative to the larger class
        bars_html = ""
        for label, count, colour in [("Slight", counts[0], "#595d6c"), ("Serious/Fatal", counts[1], "#9184d9")]:
            width = count / counts.max() * 100
            bars_html += (
                '<div style="display:flex;justify-content:space-between;font-size:13px;margin:12px 0 8px">'
                f'<span style="color:#e9e9ed">{label}</span><span style="color:#cfd3e5">{count:,} · {count / total:.1%}</span></div>'
                f'<div style="height:26px;width:{width:.1f}%;background:{colour};border-radius:4px"></div>'
            )
        st.markdown(bars_html, unsafe_allow_html=True)
        st.caption("The ~77/23 imbalance is why the model uses class-weighting rather than the raw split — see Section 15.")

    # ------------------------------------------------
    # Stage 9c: Severity rate by top predictors
    # ------------------------------------------------

    with predictors_col:
        st.markdown("#### Severity Rate by Top Predictors")
        st.caption("These three ranked highest in the SHAP feature importance — this is the pattern the model learned.")

        c1, c2, c3 = st.columns([1.4, 1.2, 0.7])

        # display the bar charts for each of the top predictors in their respective columns

        with c1.container(border=True, key="chart_card_road"):
            # display the bar chart for road type
            st.markdown("**Road Type**")
            road_type_names = {1: "Roundabout", 2: "One Way Street", 3: "Dual Carriageway",
                                6: "Single Carriageway", 7: "Slip Road", 9: "Unknown"}
            rate_by_road = df.groupby("road_type")["target"].mean().sort_values(ascending=False)
            new_index = []
            for code in rate_by_road.index:
                new_index.append(road_type_names[code])
            rate_by_road.index = new_index
            st.plotly_chart(rate_bar_fig(rate_by_road, horizontal=True, h=200), use_container_width=True, config=CHART_CONFIG)

        with c2.container(border=True, key="chart_card_speed"):
            # display the bar chart for speed limit
            st.markdown("**Speed Limit (mph)**")
            rate_by_speed = df[df["speed_limit"] > 0].groupby("speed_limit")["target"].mean() # -1 = missing
            st.plotly_chart(rate_bar_fig(rate_by_speed, horizontal=False, h=200), use_container_width=True, config=CHART_CONFIG)

        with c3.container(border=True, key="chart_card_area"):
            # display the bar chart for urban vs rural
            st.markdown("**Urban vs Rural**")
            urban_rural_df = df[(df["urban_or_rural_area"] == 1) | (df["urban_or_rural_area"] == 2)]
            rate_by_area = urban_rural_df.groupby("urban_or_rural_area")["target"].mean()
            new_index = []
            for code in rate_by_area.index:
                if code == 1:
                    new_index.append("Urban")
                else:
                    new_index.append("Rural")
            rate_by_area.index = new_index
            st.plotly_chart(rate_bar_fig(rate_by_area, horizontal=False, h=200), use_container_width=True, config=CHART_CONFIG)

# ------------------------------------------------
# Stage 10: SHAP Explorer
# ------------------------------------------------

# load the SHAP values saved by the notebook for the 5,000-row test sample, if the file exists
@st.cache_resource
def load_shap_sample():
    path = BASE_DIR / "shap_sample.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)

# top features by mean(|SHAP value|)
def importance_fig(sample, top_n=20):
    importance = pd.Series(np.abs(sample["values"]).mean(axis=0), index=sample["feature_names"])
    importance = importance.sort_values(ascending=False).head(top_n)
    labels = [FEATURE_LABELS.get(name, name) for name in importance.index]
    fig = go.Figure(go.Bar(
        orientation="h", y=labels, x=importance.values,
        marker=dict(color=NOC["accent"], opacity=scaled_opacity(importance.values, 0.4, 1.0), cornerradius=2),
        text=[f"{v:.3f}" for v in importance.values], textposition="outside",
        textfont=dict(size=12, color=NOC["muted"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    noc_layout(fig, 24 * top_n + 50)
    fig.update_layout(margin=dict(l=0, r=50, t=0, b=40), bargap=0.45)
    fig.update_xaxes(showticklabels=False, zeroline=False,
        title=dict(text="mean(|SHAP value|) — average impact on model output", font=dict(size=11, color=NOC["muted"])))
    fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    return fig

# beeswarm of the top features: one dot per collision, coloured by the feature's value
def beeswarm_fig(sample, top_n=12):
    values = sample["values"]
    data = np.asarray(sample["data"], dtype=float)
    order = np.argsort(-np.abs(values).mean(axis=0))[:top_n]
    rng = np.random.default_rng(42)

    xs, ys, colours = [], [], []
    for row, i in enumerate(order):
        feature_values = data[:, i]
        spread = feature_values.max() - feature_values.min()
        normalised = (feature_values - feature_values.min()) / spread if spread > 0 else np.full(len(feature_values), 0.5)
        xs.append(values[:, i])
        ys.append(row + rng.uniform(-0.3, 0.3, len(feature_values)))
        colours.append(normalised)

    fig = go.Figure(go.Scattergl(
        x=np.concatenate(xs), y=np.concatenate(ys), mode="markers", hoverinfo="skip",
        marker=dict(size=5, color=np.concatenate(colours), colorscale=[[0, NOC["muted"]], [1, NOC["hi"]]]),
    ))
    fig.add_vline(x=0, line_color=NOC["grid"], line_width=1)
    noc_layout(fig, 30 * top_n + 60)
    fig.update_layout(margin=dict(l=0, r=10, t=0, b=30))
    fig.update_xaxes(zeroline=False, tickfont_size=11)
    fig.update_yaxes(autorange="reversed", zeroline=False, tickvals=list(range(len(order))), ticksuffix="  ",
        ticktext=[FEATURE_LABELS.get(sample["feature_names"][i], sample["feature_names"][i]) for i in order])
    return fig

with tab_shap:

    st.header("What Drives the Model's Predictions")
    st.markdown(
        '<div style="font-size:14px;color:#cfd3e5;max-width:760px">'
        "These charts summarise the model's behaviour across the full test set, "
        "rather than a single prediction — computed via SHAP (SHapley Additive "
        "exPlanations) on the final class-weighted XGBoost model.</div>",
        unsafe_allow_html=True,
    )

    shap_sample = load_shap_sample()
    importance_col, direction_col = st.columns(2, gap="large")

    with importance_col:
        # display the SHAP feature importance chart
        st.markdown("#### Feature Importance")
        if shap_sample is not None:
            st.plotly_chart(importance_fig(shap_sample), use_container_width=True, config=CHART_CONFIG)
        else:
            st.image(str(BASE_DIR.parent / "graphs" / "shap_feature_importance.png"))
        st.caption(
            "Average magnitude of each feature's effect on the model's output, "
            "across a 5,000-row sample of the test set. Road Type, Speed Limit, and "
            "Rural Area dominate — see the Dataset Overview tab for why."
        )

    with direction_col:
        # display the SHAP dot plot chart
        st.markdown("#### Direction of Effect")
        if shap_sample is not None:
            st.plotly_chart(beeswarm_fig(shap_sample), use_container_width=True, config=CHART_CONFIG)
            st.markdown(
                '<div style="display:flex;justify-content:space-between;align-items:center;font-size:11px;color:#9397ab;margin:-8px 0 12px">'
                '<span>← toward Slight</span>'
                '<span style="display:flex;align-items:center;gap:8px">Low'
                '<span style="display:inline-block;width:56px;height:6px;border-radius:3px;background:linear-gradient(to right,#75798c,#d2cefd)"></span>'
                'High feature value</span>'
                '<span>toward Serious/Fatal →</span></div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Each dot is one collision. Bright = high value for that feature, grey = low. "
                "Position right of centre pushes the prediction toward Serious/Fatal; left "
                "pushes toward Slight. E.g. high Speed Limit (bright, right) increases risk; "
                "Junction Control: Stop Sign (bright, left) is protective."
            )
        else:
            st.image(str(BASE_DIR.parent / "graphs" / "shap_dot_plot.png"))
            st.caption(
                "Each dot is one collision. Red = high value for that feature, blue = low. "
                "Position right of centre pushes the prediction toward Serious/Fatal; left "
                "pushes toward Slight. E.g. high Speed Limit (red, right) increases risk; "
                "Junction Control: Stop Sign (red, left) is protective."
            )
