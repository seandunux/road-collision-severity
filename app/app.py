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
/* secondary (reset) button */
.stButton > button[kind="secondary"] { color: #9397ab; border-color: rgba(233,233,237,.16); }
.stButton > button[kind="secondary"]:hover { color: #cfd3e5; border-color: rgba(233,233,237,.32); }
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
/* section group headers inside the inputs card */
.field-group-label { font-size: 11px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase;
    color: #75798c; margin: 18px 0 2px; }
.field-group-label:first-child { margin-top: 0; }
/* ADDED FOR UX PASS: keyboard focus visibility, kept distinct from hover styling */
*:focus-visible { outline: 2px solid #9184d9 !important; outline-offset: 2px; }
/* ADDED FOR UX PASS: expander rows used for "About this model" / assumptions */
[data-testid="stExpander"] { border-color: rgba(233,233,237,.16) !important; }
[data-testid="stExpander"] summary { font-size: 13px !important; color: #cfd3e5 !important; }
/* the broad Inter override above also catches Streamlit's Material-Symbols icon glyphs
   (expander arrows, etc.), turning them into literal text like "keyboard_arrow_right" —
   restore the icon font specifically for those */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
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
# ADDED FOR UX PASS: a neutral alert-triangle icon, used alongside text (never colour alone) for
# "above the alert threshold" — built from primitive shapes rather than a copied icon font glyph
ICON_ALERT_TRIANGLE = ('<svg width="15" height="15" viewBox="0 0 16 16" fill="none" style="flex:none">'
    '<path d="M8 1.3 L15 14.3 H1 Z" stroke="#9184d9" stroke-width="1.3" stroke-linejoin="round"/>'
    '<rect x="7.25" y="5.6" width="1.5" height="4.2" rx="0.75" fill="#9184d9"/>'
    '<circle cx="8" cy="11.6" r="0.95" fill="#9184d9"/></svg>')
ICON_WARNING_SMALL = ('<svg width="13" height="13" viewBox="0 0 16 16" fill="none" style="flex:none">'
    '<path d="M8 1.3 L15 14.3 H1 Z" stroke="#e3b341" stroke-width="1.3" stroke-linejoin="round"/>'
    '<rect x="7.25" y="5.6" width="1.5" height="4.2" rx="0.75" fill="#e3b341"/>'
    '<circle cx="8" cy="11.6" r="0.95" fill="#e3b341"/></svg>')
# ADDED FOR UX PASS: a small neutral info icon for the data-provenance note — muted colour,
# never the sole carrier of meaning (always paired with text)
ICON_INFO = ('<svg width="13" height="13" viewBox="0 0 16 16" fill="none" style="flex:none">'
    '<circle cx="8" cy="8" r="6.7" stroke="#75798c" stroke-width="1.3"/>'
    '<rect x="7.25" y="7" width="1.5" height="5" rx="0.75" fill="#75798c"/>'
    '<circle cx="8" cy="4.6" r="0.95" fill="#75798c"/></svg>')

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
# isotonic calibrator: maps the model's raw output onto a genuinely calibrated probability
# (scale_pos_weight class-balancing distorts the raw score's scale — see notebook Section 19.4)
calibrator = save_model_data["calibrator"]
threshold = save_model_data["threshold"]
feature_names = save_model_data["feature_names"]
# ADDED FOR UX PASS: evaluation metadata attached to the bundle in notebook Section 22 — read
# here rather than hard-coded, so the "About this model" panel can't drift from the real numbers
METRICS = save_model_data.get("metrics", {})

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
    "road_type_9": "Road Type: Information unavailable",
    # junction detail (reference: Not at junction)
    "junction_detail_13": "Junction: T or Staggered",
    "junction_detail_16": "Junction: Crossroads",
    "junction_detail_17": "Junction: More Than 4 Arms",
    "junction_detail_18": "Junction: Private Drive",
    "junction_detail_19": "Junction: Other Junction",
    "junction_detail_99": "Junction: Unknown",
    # junction control (reference: -1 / not at a junction)
    "junction_control_1": "Junction Control: Traffic Signal",
    "junction_control_2": "Junction Control: Stop Sign",
    "junction_control_3": "Junction Control: Give Way / Uncontrolled",
    "junction_control_4": "Junction Control: Not Controlled",
    "junction_control_9": "Junction Control: Information unavailable",
    # pedestrian crossing (reference: -1 / data missing — "None" is its own explicit category, code 0)
    "pedestrian_crossing_0": "Pedestrian Crossing: None",
    "pedestrian_crossing_11": "Pedestrian Crossing: Pelican / Puffin / Toucan",
    "pedestrian_crossing_12": "Pedestrian Crossing: Traffic Signal Phase",
    "pedestrian_crossing_13": "Pedestrian Crossing: Footbridge / Subway",
    "pedestrian_crossing_14": "Pedestrian Crossing: Central Refuge",
    "pedestrian_crossing_15": "Pedestrian Crossing: School Patrol",
    "pedestrian_crossing_16": "Pedestrian Crossing: Zig-zag Markings",
    "pedestrian_crossing_17": "Pedestrian Crossing: Other Controlled",
    "pedestrian_crossing_99": "Pedestrian Crossing: Information unavailable",
    # light conditions (reference: Daylight)
    "light_conditions_4": "Light: Dark - Lights Lit",
    "light_conditions_5": "Light: Dark - Lights Unlit",
    "light_conditions_6": "Light: Dark - No Lighting",
    "light_conditions_7": "Light: Dark - Lighting Not Recorded",
    # weather conditions (reference: Fine)
    "weather_conditions_2": "Weather: Raining",
    "weather_conditions_3": "Weather: Snowing",
    "weather_conditions_4": "Weather: Fine / High Winds",
    "weather_conditions_5": "Weather: Rain / High Winds",
    "weather_conditions_6": "Weather: Snow / High Winds",
    "weather_conditions_7": "Weather: Fog / Mist",
    "weather_conditions_8": "Weather: Other",
    "weather_conditions_9": "Weather: Information unavailable",
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
    "special_conditions_at_site_9": "Special Conditions: Information unavailable",
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
    # trunk road (reference: -1 / not sure / not recorded)
    "trunk_road_flag_1": "Non-Trunk Road",
    "trunk_road_flag_2": "Trunk Road",
    # numeric features
    "speed_limit": "Speed Limit",
    "hour": "Hour of Day",
    "day_of_week": "Day of Week",
    "month": "Month",
}

# ------------------------------------------------------------------
# ADDED FOR UX PASS — Stage 2e: plain-language grouping for the SHAP summary
# ------------------------------------------------------------------

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# ADDED FOR UX PASS: Streamlit widgets don't reliably reset just by popping their session_state
# key — once a widget with a given key has been rendered, the browser keeps reporting its last
# value back on every rerun, re-populating that same key before the script even runs. Suffixing
# every scenario-input widget's key with a version number, and bumping the version on "Reset",
# makes each reset a genuinely new widget with no browser-side history, so it actually uses its
# coded default.
if "form_version" not in st.session_state:
    st.session_state.form_version = 0

def fkey(name):
    return f"{name}_{st.session_state.form_version}"

# each user-facing field group: (prefix(es) of its one-hot columns, label shown when none of
# those columns is active — i.e. the dropped reference category is in effect). Groups with no
# reference label here always have an explicit active column (their default is set directly
# when building the input row), so that branch is never needed for them.
CATEGORICAL_GROUPS = [
    ("Road Type", ("road_type_",), "Road Type: Roundabout"),
    ("Junction Control", ("junction_control_",), "Junction Control: Not at a junction"),
    ("Area Type", ("urban_or_rural_area_",), "Area Type: Urban"),
    ("Lighting", ("light_conditions_",), "Lighting: Daylight"),
    ("Weather", ("weather_conditions_",), "Weather: Fine"),
    ("Trunk Road", ("trunk_road_flag_",), "Trunk Road: Information unavailable"),
    ("Primary Road Class", ("first_road_class_",), "Primary Road Class: A Road"),
    ("Special Conditions", ("special_conditions_at_site_",), "Special Conditions: None"),
    ("Pedestrian Crossing", ("pedestrian_crossing_",), "Pedestrian Crossing: None"),
    # fields the form never exposes at all — bucketed as one factor so their influence is never
    # silently invisible, without inventing a specific sub-label for a combination the user didn't set
    ("Other unrecorded road/junction details",
     ("second_road_class_", "junction_detail_", "road_surface_conditions_", "carriageway_hazards_"),
     "Other unrecorded road/junction details"),
]
NUMERIC_GROUPS = ["speed_limit", "hour", "day_of_week", "month"]

def numeric_group_label(name, value):
    if name == "speed_limit":
        return f"Speed Limit: {int(value)} mph"
    if name == "hour":
        return f"Time of Day: {int(value):02d}:00"
    if name == "day_of_week":
        return f"Day of Week: {DAY_NAMES[int(value) % 7]}"
    if name == "month":
        return f"Month: {MONTH_NAMES[int(value) - 1]}"
    return name

def grouped_factors(explanation):
    """Collapses the 83 raw SHAP columns into the ~13 fields the form actually shows, summing
    each one-hot group's contributions into a single signed value with one readable label —
    e.g. all eight `weather_conditions_*` columns become one "Weather: Fine" row, matching how
    the person thinks about their own inputs rather than the model's internal encoding."""
    values = explanation.values
    data = explanation.data
    name_to_idx = {name: i for i, name in enumerate(feature_names)}

    rows = []
    for name in NUMERIC_GROUPS:
        i = name_to_idx[name]
        rows.append((numeric_group_label(name, data[i]), float(values[i])))

    for group_label, prefixes, reference_label in CATEGORICAL_GROUPS:
        idxs = [i for i, name in enumerate(feature_names) if name.startswith(prefixes)]
        total = float(values[idxs].sum())
        active = [i for i in idxs if data[i] == 1]
        if active:
            col_name = feature_names[active[0]]
            label = FEATURE_LABELS.get(col_name, col_name)
        else:
            label = reference_label
        rows.append((label, total))

    return sorted(rows, key=lambda r: -abs(r[1]))

# model suceesfully loaded: visual confirmation
st.markdown(
    '<div style="display:flex;align-items:center;gap:6px;font-size:13px;color:#9397ab;margin-top:-8px">'
    f'{ICON_CHECK_CIRCLE}<span>Model loaded successfully! The current alert threshold is {threshold:.1%}.</span></div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# ADDED FOR UX PASS — "About this model": evaluation numbers and what the model is/isn't for.
# Kept as an unobtrusive, collapsed-by-default expander rather than inline text, per the brief.
# Every figure below is read from METRICS (notebook Section 22) or the bundle's own `threshold` —
# nothing here is a separate hard-coded copy of the model's real performance.
# ------------------------------------------------------------------
with st.expander("About this model"):
    if METRICS:
        st.markdown(
            f"""
- **Data coverage:** {METRICS.get('data_coverage', 'n/a')}
- **Evaluation set:** {METRICS.get('eval_set_size', 0):,} held-out collisions, never used to fit the model or its calibration
- **ROC-AUC:** {METRICS.get('roc_auc', float('nan')):.3f}
- **PR-AUC (before calibration):** {METRICS.get('pr_auc_uncalibrated', float('nan')):.3f}
- **Brier score:** {METRICS.get('brier_score', float('nan')):.3f}
- **Average calibration gap:** under {METRICS.get('calibration_gap_pct_points', float('nan')):.1f} percentage points across the probability range
- **Current alert threshold:** {threshold:.1%} — {METRICS.get('threshold_rationale', '')}
- **Intended use:** {METRICS.get('intended_use', '')}
            """.strip()
        )
        st.caption(
            f"What \"calibrated\" means here: when the model predicts approximately 30%, about 30% of "
            f"comparable cases in the evaluation data had a serious or fatal outcome. {METRICS.get('calibration_method', '')}"
        )
    else:
        st.caption("Evaluation metadata was not found in the model bundle.")

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

            # ---- Road ----
            st.markdown('<div class="field-group-label">Road</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                speed_limit = st.selectbox("Speed Limit (mph)", options=[20, 30, 40, 50, 60, 70],
                                            index=1, key=fkey("scn_speed_limit"))
            with b:
                # "Unknown" is listed last, distinct from the ordinary conditions above it, and
                # named "Information unavailable" rather than offered as a normal scenario choice
                # — it marks incomplete police recording, not a road condition someone would
                # deliberately pick (see notebook Section 19.1 and the "About the data" note)
                road_type_label = st.selectbox(
                    "Road Type",
                    ["Roundabout", "One Way Street", "Dual Carriageway", "Single Carriageway", "Slip Road",
                     "Information unavailable"],
                    index=3, key=fkey("scn_road_type"),
                    help="\"Information unavailable\" applies the model's unrecorded-road-type category. "
                         "In the training data this coincides with less detailed incident reports, which "
                         "can lower the estimate for reasons unrelated to the road itself.",
                )
            a, b = st.columns(2)
            with a:
                area_type_label = st.selectbox("Area Type", ["Urban", "Rural"], key=fkey("scn_area_type"))
            with b:
                junction_control_label = st.selectbox(
                    "Junction Control",
                    ["Not at a junction", "Traffic Signal", "Stop Sign", "Give Way / Uncontrolled",
                     "Not Controlled", "Information unavailable"],
                    key=fkey("scn_junction_control"),
                    help="\"Not at a junction\" is the model's largest category; a small share of it "
                         "(under 2% in training data) may instead be collisions at a junction where the "
                         "control type wasn't recorded — the data can't separate the two. \"Information "
                         "unavailable\" is different: collisions where an officer marked junction control "
                         "as unrecorded were about 5x less likely to be serious/fatal, likely because "
                         "less-detailed reports tend to accompany less severe incidents — not because an "
                         "unrecorded junction is actually safer.",
                )

            # ---- Environment ----
            st.markdown('<div class="field-group-label">Environment</div>', unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                light_label = st.selectbox(
                    "Lighting Conditions",
                    ["Daylight", "Dark - Lights Lit", "Dark - Lights Unlit", "Dark - No Lighting",
                     "Dark - Lighting Not Recorded"],
                    key=fkey("scn_light"),
                    help="\"Dark - Lighting Not Recorded\" means it was dark but the lighting detail wasn't "
                         "captured at the scene — a recording gap, not a separate lighting condition.",
                )
            with b:
                weather_label = st.selectbox(
                    "Weather",
                    ["Fine", "Raining", "Snowing", "Fine / High Winds", "Rain / High Winds",
                     "Snow / High Winds", "Fog / Mist", "Other", "Information unavailable"],
                    key=fkey("scn_weather"),
                    help="\"Information unavailable\" applies the model's unrecorded-weather category, which "
                         "may reflect how the report was completed rather than the actual weather.",
                )

            # ---- Time ----
            st.markdown('<div class="field-group-label">Time</div>', unsafe_allow_html=True)
            day_of_week_label = st.selectbox(
                "Day of Week", DAY_NAMES, key=fkey("scn_day_of_week"),
            )
            hour_label = st.select_slider(
                "Time of Day", options=[f"{h:02d}:00" for h in range(24)], value="17:00", key=fkey("scn_hour"),
            )
            month_label = st.select_slider(
                "Month", options=MONTH_NAMES, value="June", key=fkey("scn_month"),
            )

            # ---- Additional road details (exposes influential fields the original form left hidden) ----
            with st.expander("Additional road details"):
                st.caption("These influence the model but weren't in the original form — set them if you know them.")
                a, b = st.columns(2)
                with a:
                    road_class_label = st.selectbox(
                        "Primary Road Class",
                        ["Motorway", "A(M) Road", "A Road", "B Road", "C Road", "Unclassified"],
                        index=2, key=fkey("scn_road_class"),
                    )
                with b:
                    trunk_road_label = st.selectbox(
                        "Trunk Road?",
                        ["Information unavailable", "Non-Trunk Road", "Trunk Road"],
                        key=fkey("scn_trunk_road"),
                        help="\"Information unavailable\" is this field's own default — most collisions in "
                             "the training data don't have a trunk-road flag recorded.",
                    )
                special_conditions_label = st.selectbox(
                    "Special Conditions at Site",
                    ["None", "Automatic Signal Out", "Automatic Signal Defective", "Roadworks",
                     "Road Surface Defective", "Oil / Diesel Spillage", "Mud on Road", "Other",
                     "Information unavailable"],
                    key=fkey("scn_special_conditions"),
                    help="\"Information unavailable\" applies the model's unrecorded category for this "
                         "field, distinct from \"None\" (no special conditions were present).",
                )
                pedestrian_crossing_label = st.selectbox(
                    "Pedestrian Crossing Nearby",
                    ["None nearby", "Pelican / Puffin / Toucan", "Traffic Signal Phase",
                     "Footbridge / Subway", "Central Refuge", "School Patrol", "Zig-zag Markings",
                     "Other Controlled Crossing", "Information unavailable"],
                    key=fkey("scn_pedestrian_crossing"),
                    help="\"Information unavailable\" is different from \"None nearby\": collisions where "
                         "the crossing type was marked unrecorded were about 5x less likely to be "
                         "serious/fatal in the training data, likely a recording pattern rather than a "
                         "genuinely safer condition.",
                )

            # ------------------------------------------------
            # Stage 5: The predict / reset buttons
            # ------------------------------------------------
            btn_predict, btn_reset = st.columns([0.65, 0.35])
            with btn_predict:
                predict_clicked = st.button("Predict Severity Risk", type="primary", use_container_width=True)
            with btn_reset:
                reset_clicked = st.button("Reset", type="secondary", use_container_width=True,
                                           help="Clear all fields back to their defaults")

        # fields not shown above still feed the model — disclosed here rather than left implicit
        with st.expander("Model assumptions for fields not shown above"):
            st.caption(
                "The model has more inputs than this form exposes. Fields below keep the dataset's "
                "most common value unless noted, so every estimate uses a complete, documented row — "
                "never information you entered."
            )
            st.markdown(
                "- **Junction layout** (e.g. crossroads, T-junction): assumed **not at a junction** "
                "(the most common case, ~50% of training records) regardless of your Junction Control "
                "choice above — a known gap, since picking \"Stop Sign\" doesn't change this field.\n"
                "- **Second road's class** (the road crossing at a junction): assumed **not applicable** "
                "— a small, less common category (~2%) kept for consistency with how the model was trained.\n"
                "- **Road surface condition** (dry/wet/icy/flooded): assumed **dry** (the most common case, "
                "~72% of training records).\n"
                "- **Carriageway hazards** (e.g. an obstruction or animal in the road): assumed **none** "
                "(~93% of training records)."
            )

    # ------------------------------------------------------------------
    # ADDED FOR UX PASS — Stage 5b: resolve every widget's current code, every run, so a
    # staleness check can compare "what's on screen" to "what the last prediction used" even on
    # a run where Predict wasn't clicked.
    # ------------------------------------------------------------------

    # road type — "Information unavailable" maps to the model's own unrecorded-road-type column
    _ROAD_TYPE_CODES = {"Roundabout": 1, "One Way Street": 2, "Dual Carriageway": 3,
                         "Single Carriageway": 6, "Slip Road": 7, "Information unavailable": 9}
    road_type_code = _ROAD_TYPE_CODES[road_type_label]

    # area type
    area_type_code = 1 if area_type_label == "Urban" else 2

    # lighting conditions — "Dark - Lighting Not Recorded" maps to the model's own unrecorded
    # lighting-detail column (still "dark", just without the specific lit/unlit/none detail)
    _LIGHT_CODES = {"Daylight": 1, "Dark - Lights Lit": 4, "Dark - Lights Unlit": 5,
                     "Dark - No Lighting": 6, "Dark - Lighting Not Recorded": 7}
    light_code = _LIGHT_CODES[light_label]

    # weather conditions — "Information unavailable" maps to the model's own unrecorded-weather column
    _WEATHER_CODES = {"Fine": 1, "Raining": 2, "Snowing": 3, "Fine / High Winds": 4,
                       "Rain / High Winds": 5, "Snow / High Winds": 6, "Fog / Mist": 7, "Other": 8,
                       "Information unavailable": 9}
    weather_code = _WEATHER_CODES[weather_label]

    # junction control — "Not at a junction" is the model's dropped reference category (-1),
    # not a placeholder for missing data (notebook Section 19 confirms ~99% of -1 rows are
    # genuinely away from a junction, not unrecorded). "Information unavailable" is a distinct,
    # explicit "unrecorded" column (code 9) — see the help text on the widget above.
    _JUNCTION_CONTROL_CODES = {"Not at a junction": -1, "Traffic Signal": 1, "Stop Sign": 2,
                                "Give Way / Uncontrolled": 3, "Not Controlled": 4,
                                "Information unavailable": 9}
    junction_control_code = _JUNCTION_CONTROL_CODES[junction_control_label]

    # day of week
    day_of_week_code = DAY_NAMES.index(day_of_week_label)

    # time of day / month, from the labelled sliders
    hour = int(hour_label.split(":")[0])
    month = MONTH_NAMES.index(month_label) + 1

    # additional road details
    _ROAD_CLASS_CODES = {"Motorway": 1, "A(M) Road": 2, "A Road": 3, "B Road": 4, "C Road": 5, "Unclassified": 6}
    road_class_code = _ROAD_CLASS_CODES[road_class_label]

    _TRUNK_ROAD_CODES = {"Information unavailable": -1, "Non-Trunk Road": 1, "Trunk Road": 2}
    trunk_road_code = _TRUNK_ROAD_CODES[trunk_road_label]

    _SPECIAL_CONDITIONS_CODES = {"None": 0, "Automatic Signal Out": 1, "Automatic Signal Defective": 2,
                                  "Roadworks": 3, "Road Surface Defective": 4, "Oil / Diesel Spillage": 5,
                                  "Mud on Road": 6, "Other": 7, "Information unavailable": 9}
    special_conditions_code = _SPECIAL_CONDITIONS_CODES[special_conditions_label]

    _PEDESTRIAN_CROSSING_CODES = {"None nearby": 0, "Pelican / Puffin / Toucan": 11,
                                   "Traffic Signal Phase": 12, "Footbridge / Subway": 13,
                                   "Central Refuge": 14, "School Patrol": 15, "Zig-zag Markings": 16,
                                   "Other Controlled Crossing": 17, "Information unavailable": 99}
    pedestrian_crossing_code = _PEDESTRIAN_CROSSING_CODES[pedestrian_crossing_label]

    # a snapshot of every input that feeds the model, used to detect "the scenario changed since
    # the last prediction" regardless of which widget changed
    current_signature = (
        speed_limit, road_type_code, area_type_code, light_code, weather_code, junction_control_code,
        day_of_week_code, hour, month, road_class_code, trunk_road_code, special_conditions_code,
        pedestrian_crossing_code,
    )

    if reset_clicked:
        # bump the key version so every scenario widget is recreated fresh next run (see fkey
        # above) — the old, now-orphaned "scn_*_<old version>" keys are simply never read again
        st.session_state.form_version += 1
        st.session_state.pop("result", None)
        st.rerun()

    if predict_clicked:

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

        # populate with one-hot encoded values for categorical features — each `if ... in
        # input_row.columns` guard is what makes picking the model's dropped reference category
        # (e.g. "Not at a junction") a no-op: no column exists for it, so the row is left at 0
        # for that whole group, which is exactly what the reference category means
        for column, code in [
            ("road_type_", road_type_code),
            ("urban_or_rural_area_", area_type_code),
            ("light_conditions_", light_code),
            ("weather_conditions_", weather_code),
            ("junction_control_", junction_control_code),
            ("first_road_class_", road_class_code),
            ("trunk_road_flag_", trunk_road_code),
            ("special_conditions_at_site_", special_conditions_code),
            ("pedestrian_crossing_", pedestrian_crossing_code),
        ]:
            col = f"{column}{code}"
            if col in input_row.columns:
                input_row[col] = 1

        # ------------------------------------------------
        # Stage 7: Make the prediction and keep the result
        # ------------------------------------------------

        # make the prediction using the trained model, then map its raw output onto a
        # calibrated probability exactly once (the raw score's scale is distorted by
        # class-weighting — see notebook Section 19.4). Nothing downstream recomputes this.
        prediction = model.predict_proba(input_row)
        probability = float(calibrator.predict([prediction[0][1]])[0])

        # calculate SHAP values for the input row
        shap_values = explainer(input_row)

        # keep the result in session state so it survives reruns and the right column can show an
        # empty state; input_signature lets a later rerun detect that the scenario has moved on
        st.session_state["result"] = {
            "probability": probability,
            "shap": shap_values[0],
            "input_signature": current_signature,
        }

    # ------------------------------------------------
    # Stage 8: The result, plain-language factors, and the detailed SHAP waterfall
    # ------------------------------------------------

    # READ: Readable labels dictionary added to 'explainer = shap.TreeExplainer(model)' in Stage 2

    # big percentage and threshold-status tag for a prediction — reframed per the UX pass:
    # neutral "above/below the alert threshold" language instead of "HIGH RISK"/"LOWER RISK",
    # the dataset baseline shown for context, and an explicit note that the threshold is a
    # precision/recall trade-off, not a safety boundary
    def result_html(p, t, dataset_prevalence):
        above = p >= t
        d = abs(p - t) * 100
        num_style = "color:#d2cefd;text-shadow:0 0 40px rgba(145,132,217,.35)" if above else "color:#e4e7f5"
        icon = ICON_ALERT_TRIANGLE if above else ICON_CHECK_CIRCLE
        tag_text = "ABOVE ALERT THRESHOLD" if above else "BELOW ALERT THRESHOLD"
        tag_style = ("border:1px solid #9184d9;color:#9184d9" if above
                     else "background:#3f424d;color:#f3f5fe")
        word = "above" if above else "below"
        baseline_html = ""
        if dataset_prevalence is not None:
            baseline_html = (
                f'<div style="font-size:13px;color:#9397ab;margin-top:10px">'
                f'Dataset average: {dataset_prevalence:.1%}</div>'
            )
        return (
            '<div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#9184d9">'
            'Estimated Probability of a Serious or Fatal Outcome</div>'
            f'<div style="font-size:112px;line-height:.95;font-weight:500;letter-spacing:-.045em;margin:14px 0 18px;{num_style}">{p*100:.1f}%</div>'
            f'<div style="display:flex;gap:8px;align-items:center">'
            f'<span style="display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:6px;'
            f'font-size:12px;letter-spacing:.08em;{tag_style}">{icon}{tag_text}</span>'
            f'<span style="font-size:14px;color:#cfd3e5">{d:.1f} points {word} the {t:.1%} alert threshold</span></div>'
            + baseline_html +
            '<div style="font-size:12px;color:#75798c;margin-top:10px;max-width:520px">'
            "The alert threshold reflects the model's chosen precision&ndash;recall trade-off; it is not a "
            "boundary between safe and dangerous conditions. This is a statistical estimate from historical "
            "patterns, not a determination of what will happen in any individual collision.</div>"
        )

    # the same block before Predict has been pressed
    def empty_result_html(t, dataset_prevalence):
        baseline_html = ""
        if dataset_prevalence is not None:
            baseline_html = (
                f'<div style="font-size:13px;color:#9397ab;margin-top:10px">'
                f'Dataset average: {dataset_prevalence:.1%}</div>'
            )
        return (
            '<div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;color:#9184d9">'
            'Estimated Probability of a Serious or Fatal Outcome</div>'
            '<div style="font-size:112px;line-height:.95;font-weight:500;letter-spacing:-.045em;margin:14px 0 18px;color:#3f424d">&mdash;%</div>'
            '<div style="display:flex;gap:10px;align-items:center">'
            '<span style="border:1px dashed #595d6c;color:#9397ab;padding:4px 12px;border-radius:6px;font-size:12px;letter-spacing:.08em">NO ESTIMATE YET</span>'
            f'<span style="font-size:14px;color:#9397ab">Alert threshold: {t:.1%}</span></div>'
            + baseline_html
        )

    # the placeholder panel that replaces the waterfall/factors before a prediction is made
    EMPTY_EXPLANATION_PANEL_HTML = (
        '<div style="border:1px dashed #3f424d;border-radius:8px;min-height:260px;padding:28px 32px;display:flex;flex-direction:column;justify-content:center;gap:8px">'
        f'<div>{ICON_CHART_BAR}</div>'
        '<div style="font-size:20px;font-weight:500;color:#cfd3e5">Why This Estimate?</div>'
        '<div style="font-size:13px;color:#9397ab;max-width:420px">Set the road and conditions on the left, then press Predict Severity Risk. '
        "The estimate and a breakdown of each factor's influence will appear here.</div></div>"
    )

    # ADDED FOR UX PASS: the Department for Transport's contributory-factors guidance covers a
    # different part of STATS19 (CF/RSF variables) than this model uses (collision-table fields
    # only) — linked here as useful background on police recording practice, not as a spec for
    # this model's specific inputs
    DFT_GUIDANCE_URL = (
        "https://www.gov.uk/government/publications/guide-to-contributory-factors-for-reported-road-casualty-statistics/"
        "guide-to-contributory-factors-for-reported-road-casualties-great-britain#strengths-and-limitations"
    )
    ABOUT_DATA_HTML = f"""
<div style="font-size:13px;color:#cfd3e5;max-width:460px;line-height:1.5">
<div style="font-weight:500;color:#e9e9ed;margin-bottom:2px">What the data represents</div>
<div style="color:#9397ab;margin-bottom:12px">This model was trained on police-reported STATS19 collision
records from Great Britain, 2020&ndash;2024. It describes patterns in recorded collisions and should not be
interpreted as representing every collision that occurred.</div>
<div style="font-weight:500;color:#e9e9ed;margin-bottom:2px">Recording effects</div>
<div style="color:#9397ab;margin-bottom:12px">Most inputs describe recorded road, environmental and time
conditions. But missing or "Information unavailable" values can also reflect how information was collected
at the scene &mdash; these recording patterns may influence the model's estimate.</div>
<div style="font-weight:500;color:#e9e9ed;margin-bottom:2px">How to interpret the result</div>
<div style="color:#9397ab;margin-bottom:12px">The displayed percentage is a calibrated model estimate for
collisions with similar recorded characteristics. It's intended for exploration and screening, not as a
prediction of what will happen in an individual event.</div>
<div style="font-weight:500;color:#e9e9ed;margin-bottom:2px">Association, not causation</div>
<div style="color:#9397ab;margin-bottom:12px">The explanation shows which recorded features influenced the
model. It does not prove that those conditions caused a serious or fatal outcome.</div>
<div style="border-top:1px solid rgba(233,233,237,.16);padding-top:10px">
<a href="{DFT_GUIDANCE_URL}" target="_blank" rel="noopener"
   style="color:#9184d9;text-decoration:underline">Department for Transport data guidance &#8599;</a>
<div style="color:#75798c;font-size:12px;margin-top:4px">This guidance focuses on contributory and road
safety factors, which are not inputs to this model, but it provides useful context on police recording
practices.</div>
</div>
</div>
"""

    # compact, always-visible provenance note + an "About the data" control that opens a
    # popover (keyboard-operable: a regular button, Tab + Enter/Space opens it)
    def render_data_context_row():
        note_col, about_col = st.columns([0.74, 0.26])
        with note_col:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:6px;font-size:12px;color:#75798c;'
                f'margin-top:2px">{ICON_INFO}<span>Based on police-reported STATS19 collisions from '
                'Great Britain, 2020&ndash;2024.</span></div>',
                unsafe_allow_html=True,
            )
        with about_col:
            with st.popover("About the data", use_container_width=True):
                st.markdown(ABOUT_DATA_HTML, unsafe_allow_html=True)

    # waterfall chart of one prediction's raw (per-column) SHAP values — unchanged from the
    # original design; kept behind "Show detailed model explanation" per the UX pass
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

    # ADDED FOR UX PASS: plain-language rows for "Factors increasing/reducing this estimate",
    # grouped by user-facing field (see grouped_factors) rather than raw one-hot columns
    def factor_rows_html(rows):
        if not rows:
            return '<div style="font-size:13px;color:#75798c">No meaningful factors.</div>'
        items = ""
        for label, value in rows:
            items += (
                '<div style="display:flex;justify-content:space-between;gap:12px;font-size:13px;'
                f'color:#cfd3e5;padding:5px 0">'
                f'<span>{label}</span><span style="color:#9397ab;flex:none">{value:+.2f}</span></div>'
            )
        return items

    dataset_prevalence = METRICS.get("dataset_prevalence")

    with right:
        result = st.session_state.get("result")

        if result is None:
            st.markdown(empty_result_html(threshold, dataset_prevalence), unsafe_allow_html=True)
            render_data_context_row()
            st.markdown(FADING_RULE + EMPTY_EXPLANATION_PANEL_HTML, unsafe_allow_html=True)
        else:
            is_stale = result.get("input_signature") != current_signature

            if is_stale:
                st.markdown(
                    '<div style="display:flex;align-items:center;gap:6px;font-size:13px;color:#e3b341;'
                    'background:rgba(227,179,65,.1);border:1px solid rgba(227,179,65,.3);border-radius:6px;'
                    f'padding:8px 12px;margin-bottom:14px">{ICON_WARNING_SMALL}<span>Scenario changed since '
                    'this estimate was produced — press Predict Severity Risk to refresh it.</span></div>',
                    unsafe_allow_html=True,
                )

            explanation = result["shap"]
            base = float(explanation.base_values)
            fx = base + explanation.values.sum()

            stale_wrap_open = '<div style="opacity:.55">' if is_stale else '<div>'
            st.markdown(
                stale_wrap_open + result_html(result["probability"], threshold, dataset_prevalence) + "</div>",
                unsafe_allow_html=True,
            )
            # compact data-context note sits beneath the result but before the explanation,
            # per the UX pass — not inside the (possibly dimmed) stale wrapper, since it's a
            # constant fact about the data rather than part of the prediction itself
            render_data_context_row()
            st.markdown(FADING_RULE, unsafe_allow_html=True)

            # ---- plain-language factors (default view) ----
            st.markdown("#### Why This Estimate?")
            factors = grouped_factors(explanation)
            increasing = [r for r in factors if r[1] > 0][:5]
            reducing = [r for r in factors if r[1] < 0][:5]

            fc1, fc2 = st.columns(2)
            with fc1:
                st.markdown('<div style="font-size:12px;color:#9397ab;margin-bottom:4px">FACTORS INCREASING THIS ESTIMATE</div>', unsafe_allow_html=True)
                st.markdown(factor_rows_html(increasing), unsafe_allow_html=True)
            with fc2:
                st.markdown('<div style="font-size:12px;color:#9397ab;margin-bottom:4px">FACTORS REDUCING THIS ESTIMATE</div>', unsafe_allow_html=True)
                st.markdown(factor_rows_html(reducing), unsafe_allow_html=True)

            st.markdown(
                '<div style="font-size:12px;color:#75798c;margin-top:10px;max-width:560px">These are model '
                "influences, not proven causes. Missing or unknown values may reflect recording practices.</div>",
                unsafe_allow_html=True,
            )

            # ---- detailed/technical view (unchanged content, now behind a toggle) ----
            with st.expander("Show detailed model explanation"):
                st.markdown(
                    f'<div style="font-size:13px;color:#9397ab;margin-top:-8px">Feature contributions, log-odds, '
                    f'E[f(x)] = {round(base, 2) + 0.0:.2f} → f(x) = {round(fx, 2) + 0.0:.2f}</div>'.replace("= -", "= −"),
                    unsafe_allow_html=True,
                )
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
        "Source: UK DfT STATS19 police-reported road collisions, Great Britain, 2020–2024. "
        "Severity is collapsed to a binary target — Serious/Fatal vs Slight.</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "These figures describe police-reported collisions. Differences in missing or unknown "
        "values may partly reflect reporting practices."
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
    # ADDED FOR UX PASS: kept visually secondary (plain expander, no icon/colour), collapsed by
    # default — a reading aid, not a warning
    with st.expander("How to read these charts"):
        st.markdown(
            '<div style="font-size:13px;color:#9397ab;max-width:640px">SHAP shows how features '
            "influenced this model's output across the evaluation sample. It explains model "
            "behaviour, not real-world causation. Strong effects for missing or unknown categories "
            "may indicate patterns in data collection rather than genuine safety effects.</div>",
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
