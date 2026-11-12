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
st.set_page_config(page_title="RoadCast — Road Collision Severity", layout="wide")

# RoadCast light-theme redesign: see design_handoff_roadcast/README.md for the full spec this
# implements. Replaces the previous "Nocturne" dark theme CSS block.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1280px; padding: 2rem 5rem 5rem; }
@media (max-width: 640px) { .block-container { padding: 1.25rem 1rem 6rem; } }
h1 { font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2 { font-size: 21px !important; font-weight: 600 !important; }
h3 { font-size: 21px !important; font-weight: 600 !important; }
/* tabs: weight + underline, not colour alone */
.stTabs [data-baseweb="tab-list"] { gap: 28px; }
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; color: #4F5663; min-height: 44px; }
.stTabs [data-baseweb="tab"] p { font-size: 15px; }
.stTabs [aria-selected="true"] { color: #16181D; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: #3A55D0; height: 2px; }
.stTabs [data-baseweb="tab-border"] { background: #E1E5EA; height: 1px; }
/* controls: 44px, 3:1 edge */
[data-baseweb="select"] > div { min-height: 44px; border-color: #8A919D; }
[data-baseweb="select"] div { font-size: 14px; }
.stButton > button { min-height: 44px; font-weight: 600; border-radius: 8px; }
.stButton > button[kind="secondary"] { border-color: #8A919D; color: #16181D; }
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] { color: #687080; font-size: 12px; }
/* quiet text buttons: "About the model" / "About the data" (any key prefixed this way, since
   "About the data" is reused with a distinct key on each tab that shows it) */
[class*="st-key-about_model_btn"] button, [class*="st-key-about_data_btn"] button {
  background: transparent !important; border: none !important; color: #3A55D0 !important;
  font-weight: 500 !important; min-height: auto !important; box-shadow: none !important; padding: 0 !important; }
[class*="st-key-about_model_btn"] button:hover, [class*="st-key-about_data_btn"] button:hover {
  text-decoration: underline; color: #2E45B0 !important; }
[class*="st-key-about_model_btn"], [class*="st-key-about_data_btn"] { display: flex; justify-content: flex-end; }
/* group heading */
.rc-group-h { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #4F5663;
              border-top: 1px solid #E1E5EA; padding-top: 20px; margin-top: 16px; margin-bottom: 4px; }
.rc-subtitle { font-size: 15px; color: #4F5663; margin-top: -8px; margin-bottom: 4px; }
/* badges */
.rc-badge { display: inline-flex; align-items: center; font-size: 13px; font-weight: 500; padding: 1px 8px;
            border-radius: 999px; margin-left: 8px; white-space: nowrap; }
.rc-badge-unrecorded { background: #EEF0F3; color: #4F5663; }
.rc-badge-changed { background: #EEF1FC; color: #2C43AE; }
.rc-hint { font-size: 13px; color: #687080; margin-top: 4px; line-height: 1.4; }
.rc-field-badges { margin-top: -4px; margin-bottom: 4px; }
/* result card */
.st-key-result_card, .st-key-result_card [data-testid="stVerticalBlockBorderWrapper"] {
  border-radius: 12px !important; border-color: #E1E5EA !important; }
/* banners / notes */
.rc-banner { display: flex; align-items: center; gap: 12px; background: #F7F8FA; border: 1px solid #C8CED7;
             border-radius: 8px; padding: 12px 16px; }
.rc-warning { background: #FFF6E5; border: 1px solid #EBC27A; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
.rc-error { background: #FEF3F2; border: 1px solid #F2B8B2; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; }
/* explanation rows */
.rc-factor-row { display: grid; grid-template-columns: 1fr 124px; gap: 12px; padding: 10px 0;
                 border-bottom: 1px solid #E1E5EA; align-items: center; }
.rc-factor-name { font-size: 15px; font-weight: 500; color: #16181D; }
.rc-factor-value { font-size: 14px; color: #4F5663; margin-top: 2px; }
.rc-track { height: 8px; border-radius: 4px; background: #E9ECF0; overflow: hidden; }
.rc-track-fill { height: 100%; border-radius: 4px; }
.rc-strength { font-size: 13px; color: #687080; margin-top: 4px; text-align: right; }
.rc-col-heading { font-size: 15px; font-weight: 600; color: #16181D; border-bottom: 1px solid #E1E5EA;
                  padding-bottom: 8px; margin-bottom: 4px; }
/* mobile: pin the primary action — fixed (not sticky), since this container is the last
   thing in its column with nothing below it to scroll past, so sticky would never engage */
@media (max-width: 640px) {
  .st-key-predict_actions { position: fixed; left: 1rem; right: 1rem; bottom: 0; z-index: 10;
                            background: rgba(255,255,255,.96); backdrop-filter: blur(4px);
                            border-top: 1px solid #E1E5EA; padding: 12px 0 16px; }
  /* Data tab: the 4-cell statistics row wraps into a 2x2 grid rather than overflowing */
  .rc-stat-row { flex-wrap: wrap; }
  .rc-stat-cell { flex: 1 1 50% !important; padding: 10px 12px !important; border-left: none !important; }
  .rc-stat-cell:nth-child(odd) { border-right: 1px solid #E1E5EA; }
  .rc-stat-cell:nth-child(n+3) { border-top: 1px solid #E1E5EA; }
}
*:focus-visible { outline: 2px solid #3A55D0 !important; outline-offset: 2px; }
[data-testid="stExpander"] { border-color: #E1E5EA !important; }
[data-testid="stExpander"] summary { font-size: 14px !important; color: #16181D !important; }
/* the broad Inter override above also catches Streamlit's Material-Symbols icon glyphs
   (expander arrows, etc.), turning them into literal text like "keyboard_arrow_right" —
   restore the icon font specifically for those */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>""", unsafe_allow_html=True)

# RoadCast light-theme chart palette and layout, shared by every Plotly chart in the app
RC = dict(text="#16181D", text2="#4F5663", muted="#687080", rule="#E1E5EA", rule_strong="#C8CED7",
          accent="#3A55D0", neutral="#8A919D", neg="#7A818D", track="#E9ECF0")

def rc_layout(fig, h):
    fig.update_layout(height=h, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=13, color=RC["text"]),
        margin=dict(l=0, r=48, t=8, b=8), showlegend=False, bargap=0.35)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=13, color=RC["text"]))
    return fig

# chart settings used for every st.plotly_chart call
CHART_CONFIG = {"displayModeBar": False}

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

try:
    save_model_data, explainer = load_model()
    MODEL_LOAD_ERROR = None
except Exception as e:
    save_model_data, explainer = None, None
    MODEL_LOAD_ERROR = str(e)

if MODEL_LOAD_ERROR:
    st.error(f"The model couldn't be loaded: {MODEL_LOAD_ERROR}")
    st.stop()

model = save_model_data["model"]
# isotonic calibrator: maps the model's raw output onto a genuinely calibrated probability
# (scale_pos_weight class-balancing distorts the raw score's scale — see notebook Section 19.4)
calibrator = save_model_data["calibrator"]
threshold = save_model_data["threshold"]
feature_names = save_model_data["feature_names"]
# evaluation metadata attached to the bundle in notebook Section 22 — read here rather than
# hard-coded, so "About the model" can't drift from the real numbers
METRICS = save_model_data.get("metrics", {})

# dataset_prevalence must stay dynamic — read from the bundle, falling back to the raw dataset
@st.cache_data
def compute_dataset_prevalence():
    try:
        df = pd.read_csv(BASE_DIR.parent / "data" / "dataset_slim.csv", usecols=["collision_severity"])
        return float(df["collision_severity"].isin([1, 2]).mean())
    except Exception:
        return None

dataset_prevalence = METRICS.get("dataset_prevalence")
if dataset_prevalence is None:
    dataset_prevalence = compute_dataset_prevalence()

# ADDED IN STAGE 8: Readable names for the model's raw column names, used on the factor rows and
# the SHAP charts (same mapping as the notebook, covering all 83 model features). Every entry is
# "Group label: value", sentence case, matching the labels shown in the Predict tab's own inputs.
FEATURE_LABELS = {
    # first road class (reference: Motorway)
    "first_road_class_2": "Primary road class: A(M) road",
    "first_road_class_3": "Primary road class: A road",
    "first_road_class_4": "Primary road class: B road",
    "first_road_class_5": "Primary road class: C road",
    "first_road_class_6": "Primary road class: Unclassified",
    # second road class (reference: -1 / not applicable)
    "second_road_class_0": "Second road: No junction",
    "second_road_class_1": "Second road: Motorway",
    "second_road_class_2": "Second road: A(M) road",
    "second_road_class_3": "Second road: A road",
    "second_road_class_4": "Second road: B road",
    "second_road_class_5": "Second road: C road",
    "second_road_class_6": "Second road: Unclassified",
    # road type (reference: Roundabout)
    "road_type_2": "Road type: One way street",
    "road_type_3": "Road type: Dual carriageway",
    "road_type_6": "Road type: Single carriageway",
    "road_type_7": "Road type: Slip road",
    "road_type_9": "Road type: Information unavailable",
    # junction detail (reference: Not at junction)
    "junction_detail_13": "Junction: T or staggered",
    "junction_detail_16": "Junction: Crossroads",
    "junction_detail_17": "Junction: More than 4 arms",
    "junction_detail_18": "Junction: Private drive",
    "junction_detail_19": "Junction: Other junction",
    "junction_detail_99": "Junction: Unknown",
    # junction control (reference: -1 / not at a junction)
    "junction_control_1": "Junction control: Traffic signal",
    "junction_control_2": "Junction control: Stop sign",
    "junction_control_3": "Junction control: Give way or uncontrolled",
    "junction_control_4": "Junction control: Not controlled",
    "junction_control_9": "Junction control: Information unavailable",
    # pedestrian crossing (reference: -1 / data missing — "None" is its own explicit category, code 0)
    "pedestrian_crossing_0": "Pedestrian crossing: None",
    "pedestrian_crossing_11": "Pedestrian crossing: Pelican, puffin or toucan",
    "pedestrian_crossing_12": "Pedestrian crossing: Traffic signal phase",
    "pedestrian_crossing_13": "Pedestrian crossing: Footbridge or subway",
    "pedestrian_crossing_14": "Pedestrian crossing: Central refuge",
    "pedestrian_crossing_15": "Pedestrian crossing: School patrol",
    "pedestrian_crossing_16": "Pedestrian crossing: Zig-zag markings",
    "pedestrian_crossing_17": "Pedestrian crossing: Other controlled",
    "pedestrian_crossing_99": "Pedestrian crossing: Information unavailable",
    # light conditions (reference: Daylight)
    "light_conditions_4": "Lighting: Dark – street lights lit",
    "light_conditions_5": "Lighting: Dark – street lights unlit",
    "light_conditions_6": "Lighting: Dark – no street lighting",
    "light_conditions_7": "Lighting: Dark – lighting not recorded",
    # weather conditions (reference: Fine)
    "weather_conditions_2": "Weather: Raining",
    "weather_conditions_3": "Weather: Snowing",
    "weather_conditions_4": "Weather: Fine with high winds",
    "weather_conditions_5": "Weather: Rain with high winds",
    "weather_conditions_6": "Weather: Snow with high winds",
    "weather_conditions_7": "Weather: Fog or mist",
    "weather_conditions_8": "Weather: Other",
    "weather_conditions_9": "Weather: Information unavailable",
    # road surface conditions (reference: Dry)
    "road_surface_conditions_2": "Road surface: Wet or damp",
    "road_surface_conditions_3": "Road surface: Snow",
    "road_surface_conditions_4": "Road surface: Frost or ice",
    "road_surface_conditions_5": "Road surface: Flood",
    "road_surface_conditions_9": "Road surface: Information unavailable",
    # special conditions (reference: -1 / data missing)
    "special_conditions_at_site_0": "Special conditions: None",
    "special_conditions_at_site_1": "Special conditions: Automatic signal out",
    "special_conditions_at_site_2": "Special conditions: Automatic signal defective",
    "special_conditions_at_site_3": "Special conditions: Roadworks",
    "special_conditions_at_site_4": "Special conditions: Road surface defective",
    "special_conditions_at_site_5": "Special conditions: Oil or diesel spillage",
    "special_conditions_at_site_6": "Special conditions: Mud on road",
    "special_conditions_at_site_7": "Special conditions: Other",
    "special_conditions_at_site_9": "Special conditions: Information unavailable",
    # carriageway hazards (reference: -1 / data missing)
    "carriageway_hazards_0": "Carriageway hazard: None",
    "carriageway_hazards_11": "Carriageway hazard: Vehicle load on road",
    "carriageway_hazards_12": "Carriageway hazard: Other object on road",
    "carriageway_hazards_13": "Carriageway hazard: Previous accident",
    "carriageway_hazards_14": "Carriageway hazard: Dog on road",
    "carriageway_hazards_15": "Carriageway hazard: Other animal",
    "carriageway_hazards_16": "Carriageway hazard: Pedestrian in road",
    "carriageway_hazards_17": "Carriageway hazard: Fallen cyclist",
    "carriageway_hazards_18": "Carriageway hazard: Other",
    "carriageway_hazards_19": "Carriageway hazard: Unknown",
    "carriageway_hazards_20": "Carriageway hazard: Not reported",
    "carriageway_hazards_21": "Carriageway hazard: Reported missing",
    "carriageway_hazards_99": "Carriageway hazard: Not recorded",
    # urban/rural (reference: Urban)
    "urban_or_rural_area_2": "Area type: Rural",
    # trunk road (reference: -1 / not sure / not recorded)
    "trunk_road_flag_1": "Trunk road: Non-trunk road",
    "trunk_road_flag_2": "Trunk road: Trunk road",
    # numeric features
    "speed_limit": "Speed limit",
    "hour": "Time of day",
    "day_of_week": "Day of week",
    "month": "Month",
}

# ------------------------------------------------------------------
# "About the model" / "About the data": st.dialog popups shown from the shared header and the
# Predict tab's result card footer. Every figure is read from METRICS/threshold — nothing here
# is a separate hard-coded copy of the model's real performance.
# ------------------------------------------------------------------

@st.dialog("About the model")
def about_model_dialog():
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

# the Department for Transport's contributory-factors guidance covers a different part of
# STATS19 (CF/RSF variables) than this model uses (collision-table fields only) — linked here as
# useful background on police recording practice, not as a spec for this model's specific inputs
DFT_GUIDANCE_URL = (
    "https://www.gov.uk/government/publications/guide-to-contributory-factors-for-reported-road-casualty-statistics/"
    "guide-to-contributory-factors-for-reported-road-casualties-great-britain#strengths-and-limitations"
)
ABOUT_DATA_HTML = f"""
<div style="font-size:14px;color:#16181D;max-width:460px;line-height:1.5">
<div style="font-weight:600;color:#16181D;margin-bottom:2px">What the data represents</div>
<div style="color:#4F5663;margin-bottom:12px">This model was trained on police-reported STATS19 collision
records from Great Britain, 2020&ndash;2024. It describes patterns in recorded collisions and should not be
interpreted as representing every collision that occurred.</div>
<div style="font-weight:600;color:#16181D;margin-bottom:2px">Recording effects</div>
<div style="color:#4F5663;margin-bottom:12px">Most inputs describe recorded road, environmental and time
conditions. But missing or "Information unavailable" values can also reflect how information was collected
at the scene &mdash; these recording patterns may influence the model's estimate.</div>
<div style="font-weight:600;color:#16181D;margin-bottom:2px">How to interpret the result</div>
<div style="color:#4F5663;margin-bottom:12px">The displayed percentage is a calibrated model estimate for
collisions with similar recorded characteristics. It's intended for exploration and screening, not as a
prediction of what will happen in an individual event.</div>
<div style="font-weight:600;color:#16181D;margin-bottom:2px">Association, not causation</div>
<div style="color:#4F5663;margin-bottom:12px">The explanation shows which recorded features influenced the
model. It does not prove that those conditions caused a serious or fatal outcome.</div>
<div style="border-top:1px solid #E1E5EA;padding-top:10px">
<a href="{DFT_GUIDANCE_URL}" target="_blank" rel="noopener"
   style="color:#3A55D0;text-decoration:underline">Department for Transport data guidance &#8599;</a>
<div style="color:#687080;font-size:13px;margin-top:4px">This guidance focuses on contributory and road
safety factors, which are not inputs to this model, but it provides useful context on police recording
practices.</div>
</div>
</div>
"""

@st.dialog("About the data")
def about_data_dialog():
    st.markdown(ABOUT_DATA_HTML, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Plain-language grouping for the SHAP summary, and the Predict tab's shared visual components
# ------------------------------------------------------------------

MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

# Streamlit widgets don't reliably reset just by popping their session_state key — once a widget
# with a given key has been rendered, the browser keeps reporting its last value back on every
# rerun, re-populating that same key before the script even runs. Suffixing every scenario-input
# widget's key with a version number, and bumping the version on "Reset", makes each reset a
# genuinely new widget with no browser-side history, so it actually uses its coded default.
if "form_version" not in st.session_state:
    st.session_state.form_version = 0

def fkey(name):
    return f"{name}_{st.session_state.form_version}"

# each user-facing field group: (prefix(es) of its one-hot columns, label shown when none of
# those columns is active — i.e. the dropped reference category is in effect). Groups with no
# reference label here always have an explicit active column (their default is set directly
# when building the input row), so that branch is never needed for them.
CATEGORICAL_GROUPS = [
    ("Road type", ("road_type_",), "Road type: Roundabout"),
    ("Junction control", ("junction_control_",), "Junction control: Not at a junction"),
    ("Area type", ("urban_or_rural_area_",), "Area type: Urban"),
    ("Lighting", ("light_conditions_",), "Lighting: Daylight"),
    ("Weather", ("weather_conditions_",), "Weather: Fine"),
    ("Road surface", ("road_surface_conditions_",), "Road surface: Dry"),
    ("Trunk road", ("trunk_road_flag_",), "Trunk road: Information unavailable"),
    ("Primary road class", ("first_road_class_",), "Primary road class: A road"),
    ("Special conditions", ("special_conditions_at_site_",), "Special conditions: None"),
    ("Pedestrian crossing", ("pedestrian_crossing_",), "Pedestrian crossing: None"),
    # fields the form never exposes at all — bucketed as one factor so their influence is never
    # silently invisible, without inventing a specific sub-label for a combination the user didn't set
    ("Other unrecorded road/junction details",
     ("second_road_class_", "junction_detail_", "carriageway_hazards_"),
     "Other unrecorded road/junction details"),
]
NUMERIC_GROUPS = ["speed_limit", "hour", "day_of_week", "month"]

def numeric_group_label(name, value):
    if name == "speed_limit":
        return f"Speed limit: {int(value)} mph"
    if name == "hour":
        return f"Time of day: {int(value):02d}:00"
    if name == "day_of_week":
        return f"Day of week: {DAY_NAMES[int(value) % 7]}"
    if name == "month":
        return f"Month: {MONTH_NAMES[int(value) - 1]}"
    return name

def grouped_factors(explanation):
    """Collapses the 83 raw SHAP columns into the ~14 fields the form actually shows, summing
    each one-hot group's contributions into a single signed value with one readable label —
    e.g. all nine `weather_conditions_*` columns become one "Weather: Fine" row, matching how
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

def status_chip(p, t):
    if p >= t:
        return ('<span style="display:inline-flex;align-items:center;gap:8px;min-height:32px;padding:0 12px;border-radius:999px;'
                'background:#EEF1FC;color:#2C43AE;box-shadow:inset 0 0 0 1px #B7C1EE;font:600 14px/1 Inter">'
                '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M6 1.5l5 8.5H1z" fill="currentColor"/></svg>'
                'Above alert threshold</span>')
    return ('<span style="display:inline-flex;align-items:center;gap:8px;min-height:32px;padding:0 12px;border-radius:999px;'
            'background:#F1F3F5;color:#3D434E;box-shadow:inset 0 0 0 1px #C8CED7;font:600 14px/1 Inter">'
            '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M6 10.5L1.6 2.5h8.8z" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/></svg>'
            'Below alert threshold</span>')

def scale_html(p, t, b, max_pct=60):
    if b is None:
        b = t
    if p is not None and p * 100 > max_pct:
        max_pct = int(np.ceil(p * 10) * 10 + 10)
    pos = lambda v: f"{min(v * 100, max_pct) / max_pct * 100:.2f}%"
    est = (f'<div style="position:absolute;top:0;left:{pos(p)};transform:translateX(-50%);'
           f'font:600 14px/20px Inter;white-space:nowrap;color:#16181D">This estimate {p:.1%}</div>') if p is not None else ""
    fill = f'<div style="position:absolute;inset:0 auto 0 0;width:{pos(p)};border-radius:5px;background:#B7C1EE"></div>' if p is not None else ""
    dot = (f'<div style="position:absolute;top:50%;left:{pos(p)};width:18px;height:18px;margin:-9px 0 0 -9px;'
           f'border-radius:50%;background:#3A55D0;border:3px solid #fff;box-shadow:0 0 0 1px #3A55D0"></div>') if p is not None else ""
    ticks = "".join(f"<span>{v}%</span>" for v in range(0, max_pct + 1, 10))
    threshold_right = t <= b
    if threshold_right:
        t_align, t_pad = "right", "padding-right:6px;transform:translateX(-100%)"
        b_align, b_pad = "left", "padding-left:6px"
    else:
        t_align, t_pad = "left", "padding-left:6px"
        b_align, b_pad = "right", "padding-right:6px;transform:translateX(-100%)"
    return f'''
<figure style="margin:0" role="img" aria-label="Scale from 0 to {max_pct} percent. Estimate {"" if p is None else f"{p:.1%}"}; alert threshold {t:.1%}; dataset baseline {b:.1%}.">
 <div style="position:relative;padding-top:{34 if p is not None else 8}px">{est}
  <div style="position:relative;height:10px;border-radius:5px;background:#E9ECF0">{fill}
   <div style="position:absolute;top:-8px;height:26px;left:{pos(t)};margin-left:-1px;border-left:2px solid #16181D"></div>
   <div style="position:absolute;top:-8px;height:26px;left:{pos(b)};margin-left:-1px;border-left:2px dashed #4F5663"></div>{dot}
  </div></div>
 <div style="position:relative;height:44px;margin-top:6px;font:13px/18px Inter">
  <div style="position:absolute;left:{pos(t)};{t_pad};text-align:{t_align};white-space:nowrap">
   <b style="display:block;color:#16181D">{t:.1%}</b><span style="color:#4F5663">Alert threshold</span></div>
  <div style="position:absolute;left:{pos(b)};{b_pad};text-align:{b_align};color:#4F5663;white-space:nowrap">
   <b style="display:block;color:#16181D">{b:.1%}</b>Dataset baseline</div>
 </div>
 <div style="display:flex;justify-content:space-between;font:13px/18px Inter;color:#687080">{ticks}</div>
</figure>'''

# ------------------------------------------------------------------
# Shared header (all tabs)
# ------------------------------------------------------------------

header_left, header_right = st.columns([0.72, 0.28])
with header_left:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-top:4px">'
        '<div style="width:24px;height:24px;border-radius:6px;background:#16181D;color:#fff;'
        'display:flex;align-items:center;justify-content:center;font:600 13px/1 Inter;flex:none">A</div>'
        '<span style="font:600 19px/1 Inter;color:#16181D">RoadCast</span></div>'
        '<div style="font-size:14px;color:#4F5663;margin:6px 0 0 34px">'
        'Explore how recorded road and environmental conditions relate to collision severity.</div>',
        unsafe_allow_html=True,
    )
with header_right:
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    if st.button("ⓘ About the model", key="about_model_btn"):
        about_model_dialog()

# --------------------------------
# Stage 3: Creating the tabs
# --------------------------------

# create the tabs for the application
tab_predict, tab_overview, tab_shap = st.tabs(["Predict", "Data", "Model insights"])

with tab_predict:

    # ------------------------------------------------
    # Stage 4: The Predict tab's input widgets
    # ------------------------------------------------

    # what the last calculated result (if any) used — drives "Changed" badges and staleness
    _stored_result = st.session_state.get("result")
    _stored_fields = _stored_result["field_labels"] if _stored_result else {}

    def group_heading(text):
        st.markdown(f'<div class="rc-group-h">{text}</div>', unsafe_allow_html=True)

    def field_badges_and_hint(value, field_name, hint=True):
        """Renders the "Changed"/"Unrecorded" badges and the unrecorded hint beneath a field.
        Returns whether this field is currently "Information unavailable" (used to decide the
        dashed-border styling)."""
        is_unrecorded = value == "Information unavailable"
        is_changed = field_name in _stored_fields and _stored_fields[field_name] != value
        badges = ""
        if is_changed:
            badges += '<span class="rc-badge rc-badge-changed">Changed</span>'
        if is_unrecorded:
            badges += '<span class="rc-badge rc-badge-unrecorded">Unrecorded</span>'
        if badges:
            st.markdown(f'<div class="rc-field-badges">{badges}</div>', unsafe_allow_html=True)
        if is_unrecorded and hint:
            st.markdown(
                '<div class="rc-hint">Not recorded. May lower the estimate for reasons unrelated '
                'to the actual conditions.</div>',
                unsafe_allow_html=True,
            )
        return is_unrecorded

    def dashed_if_unrecorded(container_key, is_unrecorded):
        if is_unrecorded:
            st.markdown(
                '<style>.st-key-' + container_key + ' [data-baseweb="select"] > div '
                '{ border-style: dashed !important; } </style>',
                unsafe_allow_html=True,
            )

    left, right = st.columns([0.37, 0.63], gap="large")

    with left:

        st.markdown("### Configure a scenario")
        st.markdown(
            '<div class="rc-subtitle">Choose the recorded conditions for a hypothetical collision.</div>',
            unsafe_allow_html=True,
        )

        # ---- Road ----
        group_heading("Road")
        a, b = st.columns(2)
        with a:
            speed_limit = st.selectbox("Speed limit", options=[20, 30, 40, 50, 60, 70],
                                        index=1, key=fkey("scn_speed_limit"),
                                        format_func=lambda v: f"{v} mph")
            field_badges_and_hint(f"{speed_limit} mph", "Speed limit", hint=False)
        with b:
            with st.container(key="field_road_type"):
                # "Information unavailable" is listed last, distinct from the ordinary conditions
                # above it — it marks incomplete police recording, not a road condition someone
                # would deliberately pick (see notebook Section 19.1 and "About the data")
                road_type_label = st.selectbox(
                    "Road type",
                    ["Roundabout", "One way street", "Dual carriageway", "Single carriageway",
                     "Slip road", "Information unavailable"],
                    index=3, key=fkey("scn_road_type"),
                    help="\"Information unavailable\" applies the model's unrecorded-road-type "
                         "category. In the training data this coincides with less detailed "
                         "incident reports, which can lower the estimate for other reasons.",
                )
                _unrec = field_badges_and_hint(road_type_label, "Road type")
            dashed_if_unrecorded("field_road_type", _unrec)

        a, b = st.columns(2)
        with a:
            area_type_label = st.selectbox("Area type", ["Urban", "Rural"], key=fkey("scn_area_type"))
            field_badges_and_hint(area_type_label, "Area type", hint=False)
        with b:
            with st.container(key="field_junction_control"):
                junction_control_label = st.selectbox(
                    "Junction control",
                    ["Not at a junction", "Traffic signal", "Stop sign", "Give way or uncontrolled",
                     "Not controlled", "Information unavailable"],
                    key=fkey("scn_junction_control"),
                    help="Most collisions are not at a junction. \"Information unavailable\" means "
                         "an officer marked junction control as unrecorded — these tend to be less "
                         "detailed reports, likely accompanying less severe incidents.",
                )
                _unrec = field_badges_and_hint(junction_control_label, "Junction control")
            dashed_if_unrecorded("field_junction_control", _unrec)

        # ---- Environment ----
        group_heading("Environment")
        a, b = st.columns(2)
        with a:
            light_label = st.selectbox(
                "Lighting",
                ["Daylight", "Dark – street lights lit", "Dark – street lights unlit",
                 "Dark – no street lighting", "Dark – lighting not recorded"],
                key=fkey("scn_light"),
                help="\"Dark – lighting not recorded\" means it was dark but the lighting detail "
                     "wasn't captured at the scene — a recording gap, not a lighting condition.",
            )
            field_badges_and_hint(light_label, "Lighting", hint=False)
        with b:
            with st.container(key="field_weather"):
                weather_label = st.selectbox(
                    "Weather",
                    ["Fine", "Raining", "Snowing", "Fine with high winds", "Rain with high winds",
                     "Snow with high winds", "Fog or mist", "Other", "Information unavailable"],
                    key=fkey("scn_weather"),
                    help="\"Information unavailable\" applies the model's unrecorded-weather "
                         "category, which may reflect how the report was completed.",
                )
                _unrec = field_badges_and_hint(weather_label, "Weather")
            dashed_if_unrecorded("field_weather", _unrec)

        a, b = st.columns(2)
        with a:
            with st.container(key="field_road_surface"):
                # road_surface_conditions_* was previously fixed to Dry under "Values the model
                # assumes" — it's exposed here now, per the redesign brief
                road_surface_label = st.selectbox(
                    "Road surface",
                    ["Dry", "Wet or damp", "Snow", "Frost or ice", "Flood", "Information unavailable"],
                    key=fkey("scn_road_surface"),
                    help="The recorded state of the road surface at the time of the collision.",
                )
                _unrec = field_badges_and_hint(road_surface_label, "Road surface")
            dashed_if_unrecorded("field_road_surface", _unrec)
        with b:
            st.empty()

        # ---- Time ----
        group_heading("Time")
        a, b = st.columns(2)
        with a:
            day_of_week_label = st.selectbox("Day of week", DAY_NAMES, key=fkey("scn_day_of_week"))
            field_badges_and_hint(day_of_week_label, "Day of week", hint=False)
        with b:
            month_label = st.selectbox("Month", MONTH_NAMES, index=5, key=fkey("scn_month"))
            field_badges_and_hint(month_label, "Month", hint=False)
        hour_label = st.select_slider(
            "Time of day", options=[f"{h:02d}:00" for h in range(24)], value="17:00", key=fkey("scn_hour"),
        )
        field_badges_and_hint(hour_label, "Time of day", hint=False)

        # ---- expanders ----
        # read the previous run's values (if any) to build each expander's one-line summary
        # label before the expander — and the widgets inside it — are (re)created this run
        _road_class_prev = st.session_state.get(fkey("scn_road_class"), "A road")
        _trunk_road_prev = st.session_state.get(fkey("scn_trunk_road"), "Information unavailable")
        _special_conditions_prev = st.session_state.get(fkey("scn_special_conditions"), "None")
        _pedestrian_crossing_prev = st.session_state.get(fkey("scn_pedestrian_crossing"), "None nearby")

        _summary_parts = [
            _road_class_prev,
            "Trunk road unrecorded" if _trunk_road_prev == "Information unavailable" else _trunk_road_prev,
            "No special conditions" if _special_conditions_prev == "None" else _special_conditions_prev,
            "No crossing nearby" if _pedestrian_crossing_prev == "None nearby" else _pedestrian_crossing_prev,
        ]
        more_road_summary = " · ".join(_summary_parts)

        with st.expander(f"More road details — {more_road_summary}"):
            a, b = st.columns(2)
            with a:
                road_class_label = st.selectbox(
                    "Primary road class",
                    ["Motorway", "A(M) road", "A road", "B road", "C road", "Unclassified"],
                    index=2, key=fkey("scn_road_class"),
                )
            with b:
                with st.container(key="field_trunk_road"):
                    trunk_road_label = st.selectbox(
                        "Trunk road",
                        ["Information unavailable", "Non-trunk road", "Trunk road"],
                        key=fkey("scn_trunk_road"),
                        help="\"Information unavailable\" is this field's own default — most "
                             "collisions in the training data don't have this flag recorded.",
                    )
                    _unrec = field_badges_and_hint(trunk_road_label, "Trunk road")
                dashed_if_unrecorded("field_trunk_road", _unrec)
            with st.container(key="field_special_conditions"):
                special_conditions_label = st.selectbox(
                    "Special conditions at site",
                    ["None", "Automatic signal out", "Automatic signal defective", "Roadworks",
                     "Road surface defective", "Oil or diesel spillage", "Mud on road", "Other",
                     "Information unavailable"],
                    key=fkey("scn_special_conditions"),
                    help="\"Information unavailable\" applies the model's unrecorded category for "
                         "this field, distinct from \"None\" (no special conditions present).",
                )
                _unrec = field_badges_and_hint(special_conditions_label, "Special conditions")
            dashed_if_unrecorded("field_special_conditions", _unrec)
            with st.container(key="field_pedestrian_crossing"):
                pedestrian_crossing_label = st.selectbox(
                    "Pedestrian crossing nearby",
                    ["None nearby", "Pelican, puffin or toucan", "Traffic signal phase",
                     "Footbridge or subway", "Central refuge", "School patrol", "Zig-zag markings",
                     "Other controlled crossing", "Information unavailable"],
                    key=fkey("scn_pedestrian_crossing"),
                    help="\"Information unavailable\" is different from \"None nearby\": these "
                         "were about 5x less likely to be serious/fatal in the training data, "
                         "likely a recording pattern rather than a genuinely safer condition.",
                )
                _unrec = field_badges_and_hint(pedestrian_crossing_label, "Pedestrian crossing")
            dashed_if_unrecorded("field_pedestrian_crossing", _unrec)

        with st.expander("Values the model assumes — Junction layout, second road and carriageway hazards"):
            st.caption(
                "The model has more inputs than this form exposes. Fields below keep the "
                "dataset's most common value, so every estimate uses a complete, documented "
                "row — never information you entered."
            )
            st.markdown(
                "- **Junction layout** (e.g. crossroads, T-junction): assumed **not at a junction** "
                "(the most common case, ~50% of training records) regardless of your Junction "
                "control choice above — a known gap, since picking \"Stop sign\" doesn't change "
                "this field.\n"
                "- **Second road's class** (the road crossing at a junction): assumed **not "
                "applicable** — a small, less common category (~2%) kept for consistency with "
                "how the model was trained.\n"
                "- **Carriageway hazards** (e.g. an obstruction or animal in the road): assumed "
                "**none** (~93% of training records)."
            )

        # a snapshot of every field as currently displayed — the single source of truth for
        # "has the scenario changed since the last calculation", and for the "Changed" badges above
        FIELD_DISPLAY = {
            "Speed limit": f"{speed_limit} mph",
            "Road type": road_type_label,
            "Area type": area_type_label,
            "Junction control": junction_control_label,
            "Lighting": light_label,
            "Weather": weather_label,
            "Road surface": road_surface_label,
            "Day of week": day_of_week_label,
            "Time of day": hour_label,
            "Month": month_label,
            "Primary road class": road_class_label,
            "Trunk road": trunk_road_label,
            "Special conditions": special_conditions_label,
            "Pedestrian crossing": pedestrian_crossing_label,
        }
        is_stale = _stored_result is not None and _stored_fields != FIELD_DISPLAY
        btn_label = "Calculate again" if is_stale else "Calculate estimate"

        # ------------------------------------------------
        # Stage 5: The predict / reset buttons
        # ------------------------------------------------
        with st.container(key="predict_actions"):
            btn_predict, btn_reset = st.columns([0.62, 0.38])
            with btn_predict:
                predict_clicked = st.button(btn_label, type="primary", use_container_width=True)
            with btn_reset:
                reset_clicked = st.button("Reset scenario", type="secondary", use_container_width=True,
                                           help="Clear all fields back to their defaults")

    # ------------------------------------------------------------------
    # Stage 5b: resolve every widget's current code, every run, so predicting and the staleness
    # check both use the same values regardless of which widget changed
    # ------------------------------------------------------------------

    # road type — "Information unavailable" maps to the model's own unrecorded-road-type column
    _ROAD_TYPE_CODES = {"Roundabout": 1, "One way street": 2, "Dual carriageway": 3,
                         "Single carriageway": 6, "Slip road": 7, "Information unavailable": 9}
    road_type_code = _ROAD_TYPE_CODES[road_type_label]

    # area type
    area_type_code = 1 if area_type_label == "Urban" else 2

    # lighting conditions — "Dark – lighting not recorded" maps to the model's own unrecorded
    # lighting-detail column (still "dark", just without the specific lit/unlit/none detail)
    _LIGHT_CODES = {"Daylight": 1, "Dark – street lights lit": 4, "Dark – street lights unlit": 5,
                     "Dark – no street lighting": 6, "Dark – lighting not recorded": 7}
    light_code = _LIGHT_CODES[light_label]

    # weather conditions — "Information unavailable" maps to the model's own unrecorded-weather column
    _WEATHER_CODES = {"Fine": 1, "Raining": 2, "Snowing": 3, "Fine with high winds": 4,
                       "Rain with high winds": 5, "Snow with high winds": 6, "Fog or mist": 7,
                       "Other": 8, "Information unavailable": 9}
    weather_code = _WEATHER_CODES[weather_label]

    # road surface conditions — "Dry" is the model's dropped reference category (no column)
    _ROAD_SURFACE_CODES = {"Dry": 1, "Wet or damp": 2, "Snow": 3, "Frost or ice": 4, "Flood": 5,
                            "Information unavailable": 9}
    road_surface_code = _ROAD_SURFACE_CODES[road_surface_label]

    # junction control — "Not at a junction" is the model's dropped reference category (-1),
    # not a placeholder for missing data (notebook Section 19 confirms ~99% of -1 rows are
    # genuinely away from a junction, not unrecorded). "Information unavailable" is a distinct,
    # explicit "unrecorded" column (code 9) — see the help text on the widget above.
    _JUNCTION_CONTROL_CODES = {"Not at a junction": -1, "Traffic signal": 1, "Stop sign": 2,
                                "Give way or uncontrolled": 3, "Not controlled": 4,
                                "Information unavailable": 9}
    junction_control_code = _JUNCTION_CONTROL_CODES[junction_control_label]

    # day of week
    day_of_week_code = DAY_NAMES.index(day_of_week_label)

    # time of day / month, from the labelled widgets
    hour = int(hour_label.split(":")[0])
    month = MONTH_NAMES.index(month_label) + 1

    # additional road details
    _ROAD_CLASS_CODES = {"Motorway": 1, "A(M) road": 2, "A road": 3, "B road": 4, "C road": 5, "Unclassified": 6}
    road_class_code = _ROAD_CLASS_CODES[road_class_label]

    _TRUNK_ROAD_CODES = {"Information unavailable": -1, "Non-trunk road": 1, "Trunk road": 2}
    trunk_road_code = _TRUNK_ROAD_CODES[trunk_road_label]

    _SPECIAL_CONDITIONS_CODES = {"None": 0, "Automatic signal out": 1, "Automatic signal defective": 2,
                                  "Roadworks": 3, "Road surface defective": 4, "Oil or diesel spillage": 5,
                                  "Mud on road": 6, "Other": 7, "Information unavailable": 9}
    special_conditions_code = _SPECIAL_CONDITIONS_CODES[special_conditions_label]

    _PEDESTRIAN_CROSSING_CODES = {"None nearby": 0, "Pelican, puffin or toucan": 11,
                                   "Traffic signal phase": 12, "Footbridge or subway": 13,
                                   "Central refuge": 14, "School patrol": 15, "Zig-zag markings": 16,
                                   "Other controlled crossing": 17, "Information unavailable": 99}
    pedestrian_crossing_code = _PEDESTRIAN_CROSSING_CODES[pedestrian_crossing_label]

    if reset_clicked:
        # bump the key version so every scenario widget is recreated fresh next run (see fkey
        # above) — the old, now-orphaned "scn_*_<old version>" keys are simply never read again
        st.session_state.form_version += 1
        st.session_state.pop("result", None)
        st.session_state.pop("predict_error", None)
        st.rerun()

    # ------------------------------------------------
    # Stage 6: The result card
    # ------------------------------------------------

    def waterfall_fig(explanation, top_n=11):
        values = explanation.values
        base = float(explanation.base_values)
        order = np.argsort(-np.abs(values)) # largest contribution first
        top = order[:top_n]
        rest = order[top_n:]

        def display_label(i):
            raw_name = feature_names[i]
            if raw_name in NUMERIC_GROUPS:
                full_label = numeric_group_label(raw_name, explanation.data[i])
            else:
                full_label = FEATURE_LABELS.get(raw_name, raw_name)
            if ":" in full_label:
                group, val = full_label.split(":", 1)
                return f"{group.strip()} · {val.strip()}"
            return full_label

        # rows are built bottom to top, so the running total starts at the base value
        labels, deltas = [], []
        if len(rest) > 0:
            labels.append(f"{len(rest)} other features")
            deltas.append(values[rest].sum())
        for i in top[::-1]:
            labels.append(display_label(i))
            deltas.append(values[i])

        starts = base + np.concatenate([[0], np.cumsum(deltas)[:-1]])
        colours = [RC["accent"] if d >= 0 else RC["neg"] for d in deltas]
        rows = list(range(len(labels)))

        fig = go.Figure(go.Bar(
            orientation="h", y=rows, x=deltas, base=starts,
            marker=dict(color=colours, cornerradius=2),
            hovertemplate="%{customdata}<extra></extra>",
            customdata=[f"{d:+.3f}" for d in deltas],
        ))
        fig.add_vline(x=base, line_color=RC["rule_strong"], line_width=1)

        # contribution values in a column on the right, e.g. +0.31 / −0.12
        for row, d in zip(rows, deltas):
            fig.add_annotation(x=1, xref="paper", y=row, xanchor="left", xshift=12, showarrow=False,
                text=f"{d:+.2f}".replace("-", "−"),
                font=dict(size=13, color=RC["text"]))

        rc_layout(fig, 25 * len(labels) + 30)
        fig.update_layout(margin=dict(l=0, r=56, t=5, b=5), bargap=0.45)
        fig.update_xaxes(showticklabels=False, zeroline=False)
        fig.update_yaxes(tickvals=rows, ticktext=labels, ticksuffix="  ")
        return fig

    with right:
        with st.container(border=True, key="result_card"):
            stored_result = _stored_result
            banner_clicked = False
            error_retry_clicked = False

            if stored_result is not None and is_stale:
                changed_bits = []
                for name, new_val in FIELD_DISPLAY.items():
                    old_val = stored_result["field_labels"].get(name)
                    if old_val is not None and old_val != new_val:
                        changed_bits.append(f"{name} changed from {old_val} to {new_val}")
                change_text = "; ".join(changed_bits) + ". Calculate again to update it."
                banner_cols = st.columns([0.78, 0.22])
                with banner_cols[0]:
                    st.markdown(
                        '<div class="rc-banner">↻<div>'
                        '<div style="font-weight:600;font-size:15px;color:#16181D">This estimate is out of date</div>'
                        f'<div style="font-size:14px;color:#4F5663">{change_text}</div></div></div>',
                        unsafe_allow_html=True,
                    )
                with banner_cols[1]:
                    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
                    banner_clicked = st.button("Calculate again", type="primary", key="banner_recalc",
                                                use_container_width=True)
                st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

            if st.session_state.get("predict_error"):
                st.markdown(
                    '<div class="rc-error">'
                    '<div style="font-weight:600;color:#B42318;font-size:15px">⊗ The estimate couldn\'t be calculated</div>'
                    '<div style="font-size:14px;color:#7A2E27;margin-top:4px">The model didn\'t return a result. '
                    'Your scenario hasn\'t changed, so you can try again. If this keeps happening, reload the page.</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
                error_retry_clicked = st.button("↻ Try again", key="error_retry")
                with st.expander("Technical details"):
                    st.code(st.session_state.get("predict_error", ""))

            do_predict = predict_clicked or banner_clicked or error_retry_clicked

            if do_predict:
                try:
                    with st.spinner("Calculating estimate…"):
                        # ------------------------------------------------
                        # Stage 7: Building the model's input and predicting
                        # ------------------------------------------------
                        input_row = pd.DataFrame(0, index=[0], columns=feature_names)
                        input_row["speed_limit"] = speed_limit
                        input_row["hour"] = hour
                        input_row["day_of_week"] = day_of_week_code
                        input_row["month"] = month

                        # populate one-hot encoded values — each `if ... in input_row.columns`
                        # guard is what makes picking the model's dropped reference category
                        # (e.g. "Not at a junction") a no-op: no column exists for it, so the row
                        # is left at 0 for that whole group, which is exactly what it means
                        for column, code in [
                            ("road_type_", road_type_code),
                            ("urban_or_rural_area_", area_type_code),
                            ("light_conditions_", light_code),
                            ("weather_conditions_", weather_code),
                            ("road_surface_conditions_", road_surface_code),
                            ("junction_control_", junction_control_code),
                            ("first_road_class_", road_class_code),
                            ("trunk_road_flag_", trunk_road_code),
                            ("special_conditions_at_site_", special_conditions_code),
                            ("pedestrian_crossing_", pedestrian_crossing_code),
                        ]:
                            col = f"{column}{code}"
                            if col in input_row.columns:
                                input_row[col] = 1

                        # make the prediction, then map its raw output onto a calibrated
                        # probability exactly once (the raw score's scale is distorted by
                        # class-weighting — see notebook Section 19.4). Nothing downstream
                        # recomputes this.
                        prediction = model.predict_proba(input_row)
                        probability = float(calibrator.predict([prediction[0][1]])[0])
                        shap_values = explainer(input_row)

                    st.session_state["result"] = {
                        "probability": probability,
                        "shap": shap_values[0],
                        "field_labels": dict(FIELD_DISPLAY),
                    }
                    st.session_state.pop("predict_error", None)
                except Exception as e:
                    st.session_state["predict_error"] = str(e)
                # the banner/error block above was drawn using this run's *pre-calculation*
                # state (it has to be, since its own button is part of do_predict) — rerunning
                # unconditionally here means the very next run renders the card fresh, so a
                # just-recalculated result never flashes a stale banner over its own new number
                st.rerun()

            # ---- render the card body: error / empty / result ----
            if st.session_state.get("predict_error"):
                pass # the error alert above is the whole card body
            elif stored_result is None:
                st.markdown("## Your estimate will appear here")
                st.markdown(
                    '<div style="font-size:15px;color:#4F5663;margin-top:-8px;margin-bottom:16px">'
                    'Configure the recorded road, environmental, and time conditions, then '
                    'calculate the estimate.</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(scale_html(None, threshold, dataset_prevalence), unsafe_allow_html=True)
                st.markdown(
                    '<div style="font-size:13px;color:#687080;margin-top:14px">'
                    'Based on police-reported STATS19 collisions from Great Britain, 2020–2024.</div>',
                    unsafe_allow_html=True,
                )
            else:
                probability = stored_result["probability"]
                explanation = stored_result["shap"]

                missing_fields = [name for name in ("Road type", "Junction control", "Weather", "Road surface")
                                   if stored_result["field_labels"].get(name) == "Information unavailable"]

                # a single st.container is the only way to make several separate st.markdown
                # calls dim together — an opening <div> in one markdown call and a closing
                # </div> in a later one never actually nest (each st.markdown renders as its
                # own self-contained HTML fragment), so the opacity must be scoped via CSS
                # against this container's own st-key class instead
                with st.container(key="result_body"):
                    if missing_fields:
                        parts = [missing_fields[0]] + [f[0].lower() + f[1:] for f in missing_fields[1:]]
                        field_text = parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
                        st.markdown(
                            '<div class="rc-warning">'
                            f'<div style="font-weight:600;color:#7A4600;font-size:15px">⚠ {len(missing_fields)} '
                            f'input{"s" if len(missing_fields) != 1 else ""} '
                            f'{"are" if len(missing_fields) != 1 else "is"} set to Information unavailable</div>'
                            f'<div style="font-size:14px;color:#7A4600;margin-top:4px">{field_text}. In the '
                            'training data, unrecorded values often accompany less detailed reports, which can '
                            'lower the estimate for reasons unrelated to the road or weather.</div></div>',
                            unsafe_allow_html=True,
                        )

                    kicker_badge = ' <span class="rc-badge rc-badge-unrecorded">Outdated</span>' if is_stale else ""
                    st.markdown(
                        '<div style="font-size:15px;font-weight:500;color:#4F5663">Estimated probability of a '
                        f'serious or fatal outcome{kicker_badge}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        '<div style="display:flex;align-items:baseline;gap:16px;flex-wrap:wrap;margin:6px 0 18px">'
                        f'<div style="font-size:72px;font-weight:600;letter-spacing:-0.035em;color:#16181D;'
                        f'font-variant-numeric:tabular-nums;line-height:1">{probability:.1%}</div>'
                        f'<div>{status_chip(probability, threshold)}</div></div>',
                        unsafe_allow_html=True,
                    )

                    t_diff = abs(probability - threshold) * 100
                    t_word = "above" if probability >= threshold else "below"
                    comp_cols = st.columns(2)
                    with comp_cols[0]:
                        st.markdown(
                            '<div style="border-top:1px solid #E1E5EA;padding-top:12px">'
                            '<div style="font-size:14px;color:#4F5663">│ Alert threshold</div>'
                            f'<div style="font-size:20px;font-weight:600;color:#16181D;margin-top:2px">{threshold:.1%}</div>'
                            f'<div style="font-size:13px;color:#687080;margin-top:2px">{t_diff:.1f} points {t_word}</div></div>',
                            unsafe_allow_html=True,
                        )
                    with comp_cols[1]:
                        if dataset_prevalence is not None:
                            b_diff = abs(probability - dataset_prevalence) * 100
                            b_word = "above" if probability >= dataset_prevalence else "below"
                            st.markdown(
                                '<div style="border-top:1px dashed #C8CED7;padding-top:12px">'
                                '<div style="font-size:14px;color:#4F5663">┊ Dataset baseline</div>'
                                f'<div style="font-size:20px;font-weight:600;color:#16181D;margin-top:2px">{dataset_prevalence:.1%}</div>'
                                f'<div style="font-size:13px;color:#687080;margin-top:2px">{b_diff:.1f} points {b_word}</div></div>',
                                unsafe_allow_html=True,
                            )

                    st.markdown(
                        '<div style="margin-top:18px">' + scale_html(probability, threshold, dataset_prevalence) + '</div>',
                        unsafe_allow_html=True,
                    )

                    st.markdown(
                        '<div style="font-size:14px;color:#4F5663;margin-top:14px">The alert threshold reflects '
                        "the model's selected precision–recall trade-off. It is not a boundary between safe and "
                        'dangerous conditions.</div>',
                        unsafe_allow_html=True,
                    )
                if is_stale:
                    st.markdown('<style>.st-key-result_body { opacity: 0.5; }</style>', unsafe_allow_html=True)

                st.markdown('<div style="height:1px;background:#E1E5EA;margin:20px 0 14px"></div>', unsafe_allow_html=True)
                footer_cols = st.columns([0.7, 0.3])
                with footer_cols[0]:
                    st.markdown(
                        '<div style="font-size:13px;color:#687080;padding-top:6px">Based on police-reported '
                        'STATS19 collisions from Great Britain, 2020–2024.</div>',
                        unsafe_allow_html=True,
                    )
                with footer_cols[1]:
                    if st.button("About the data", key="about_data_btn"):
                        about_data_dialog()

        # ------------------------------------------------
        # Stage 8: Explanation, below the card (not boxed)
        # ------------------------------------------------
        if stored_result is not None and not st.session_state.get("predict_error"):
            st.markdown("## What influenced this estimate", unsafe_allow_html=False)
            st.markdown(
                '<div style="font-size:15px;color:#4F5663;margin-top:-10px">These are model influences, not '
                'proven causes. Missing or unknown values may reflect recording practices.</div>',
                unsafe_allow_html=True,
            )

            factors = grouped_factors(stored_result["shap"])
            increasing = [r for r in factors if r[1] > 0]
            reducing = [r for r in factors if r[1] < 0]

            def factor_rows_html(rows):
                if not rows:
                    return '<div style="font-size:14px;color:#687080;padding:10px 0">No meaningful factors.</div>'
                out = ""
                for label, value in rows:
                    if ":" in label:
                        name, used = label.split(":", 1)
                        name, used = name.strip(), used.strip()
                    else:
                        name, used = label, ""
                    frac = min(abs(value) / 0.45, 1.0) * 100
                    colour = RC["accent"] if value > 0 else RC["neg"]
                    strength = "Strong" if abs(value) >= 0.15 else ("Moderate" if abs(value) >= 0.07 else "Slight")
                    unrecorded_tag = (' <span class="rc-badge rc-badge-unrecorded">Unrecorded</span>'
                                       if used == "Information unavailable" else "")
                    out += (
                        '<div class="rc-factor-row">'
                        f'<div><div class="rc-factor-name">{name}</div>'
                        f'<div class="rc-factor-value">{used}{unrecorded_tag}</div></div>'
                        '<div><div class="rc-track"><div class="rc-track-fill" '
                        f'style="width:{frac:.0f}%;background:{colour}"></div></div>'
                        f'<div class="rc-strength">{strength}</div></div></div>'
                    )
                return out

            fc1, fc2 = st.columns(2, gap="large")
            with fc1:
                st.markdown('<div class="rc-col-heading">↑ Factors increasing this estimate</div>', unsafe_allow_html=True)
                st.markdown(factor_rows_html(increasing), unsafe_allow_html=True)
            with fc2:
                st.markdown('<div class="rc-col-heading">↓ Factors reducing this estimate</div>', unsafe_allow_html=True)
                st.markdown(factor_rows_html(reducing), unsafe_allow_html=True)

            with st.expander("View technical explanation"):
                st.markdown(
                    '<div style="font-size:14px;color:#4F5663;margin-bottom:8px">SHAP contributions on the '
                    "model's log-odds scale. The raw model output is in log-odds; the percentage above is "
                    "produced from it by a separate calibration step, so the two scales don't map one-to-one."
                    '</div>',
                    unsafe_allow_html=True,
                )
                st.plotly_chart(waterfall_fig(stored_result["shap"]), use_container_width=True, config=CHART_CONFIG)

# ------------------------------------------------
# Stage 9: Data tab
# ------------------------------------------------

# load the dataset once and reuse it across reruns
@st.cache_data
def load_dataset():
    # slim copy of the full dataset containing only the needed columns
    columns_needed = ["collision_severity", "road_type", "speed_limit", "urban_or_rural_area",
                       "collision_year", "light_conditions"]

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

# a severity-rate bar chart shared by every "story row" below: one bar per category, the
# highlighted category (if any) in the accent colour, the rest neutral, an "Information
# unavailable"/"not recorded" category (if any) drawn hatched and always kept last, and a
# dashed baseline line for the dataset-wide rate
def data_bar_fig(categories, rates, counts, *, horizontal, highlight=None, hatch=None,
                  baseline=None, h=220, axis_max=None):
    colours, shapes = [], []
    for cat in categories:
        if hatch is not None and cat == hatch:
            colours.append("#FFFFFF")
            shapes.append("/")
        elif cat == highlight:
            colours.append(RC["accent"])
            shapes.append("")
        else:
            colours.append(RC["neutral"])
            shapes.append("")
    # a vertical bar's label sits above a narrow column, so it only has room for the rate —
    # the count still has room beside a horizontal bar, and is always in the data-table toggle
    if horizontal:
        text = [f"<b>{r:.1%}</b>  {c:,}" for r, c in zip(rates, counts)]
    else:
        text = [f"<b>{r:.1%}</b>" for r in rates]
    marker = dict(color=colours, line=dict(color=RC["neutral"], width=1),
                  pattern=dict(shape=shapes, fgcolor=RC["rule_strong"], size=6))
    if horizontal:
        bar = go.Bar(orientation="h", y=categories, x=rates, text=text, marker=marker)
    else:
        bar = go.Bar(x=categories, y=rates, text=text, marker=marker)
    # a vertical bar's label also needs to be narrower than its own column at the smallest
    # supported width (6 categories on a ~360px mobile chart) — 13px works for a horizontal
    # bar's full-row width, but a vertical bar's percentage-only label needs to run smaller
    bar.update(textposition="outside", textfont=dict(size=13 if horizontal else 11, color=RC["text"]),
               cliponaxis=False, hovertemplate="%{text}<extra></extra>")
    fig = rc_layout(go.Figure(bar), h)
    fig.update_layout(margin=dict(l=0, r=16, t=16, b=4))
    # "outside" bar-end labels need real headroom above the tallest bar, or closely-valued short
    # bars end up with overlapping labels — the caption below each chart already explains the
    # dashed line, so this only draws the line itself, with no annotation competing for space
    axis_max = axis_max or (max(rates) * 1.45 if len(rates) else 1)
    if baseline is not None:
        if horizontal:
            fig.add_vline(x=baseline, line_dash="dash", line_width=1.5, line_color=RC["text2"])
        else:
            fig.add_hline(y=baseline, line_dash="dash", line_width=1.5, line_color=RC["text2"])
    if horizontal:
        fig.update_xaxes(showticklabels=False, zeroline=False, range=[0, axis_max])
        fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    else:
        fig.update_yaxes(showticklabels=False, zeroline=False, range=[0, axis_max])
        fig.update_xaxes(type="category")
    return fig

# one "story row": a left-hand takeaway (with an optional data-table toggle) beside a chart
def story_row(question, takeaway, support, fig, table_df, table_key, caption):
    left, right = st.columns([0.3, 0.7], gap="large")
    with left:
        st.markdown(f"#### {question}")
        st.markdown(f'<div style="font-size:17px;color:#16181D">{takeaway}</div>', unsafe_allow_html=True)
        if support:
            st.markdown(f'<div style="font-size:14px;color:#4F5663;margin-top:8px">{support}</div>',
                        unsafe_allow_html=True)
        show_table = st.toggle("View data table", key=table_key)
    with right:
        st.plotly_chart(fig, use_container_width=True, config=CHART_CONFIG)
        st.caption(caption)
    if show_table:
        st.dataframe(table_df, use_container_width=True, hide_index=True)
    st.markdown('<div style="height:1px;background:#E1E5EA;margin:8px 0 32px"></div>', unsafe_allow_html=True)

with tab_overview:

    df = load_dataset()

    # ------------------------------------------------
    # Stage 9a: Header and summary statistics
    # ------------------------------------------------

    st.markdown("# Understand the data")
    st.markdown(
        '<div style="font-size:15px;color:#4F5663;margin-top:-8px">Explore the police-reported collision '
        'records used to train and evaluate the model.</div>',
        unsafe_allow_html=True,
    )

    n_raw = len(df)
    n_serious = int(df["target"].sum())
    serious_rate = df["target"].mean()
    year_min, year_max = df["collision_year"].min(), df["collision_year"].max()
    n_cleaned = METRICS.get("cleaned_rows", 503022) # dropped during notebook cleaning (Sections 4–7); not derivable from this slim extract

    stats = [
        (f"{n_raw:,}", "Raw collision records", "#16181D"),
        (f"{serious_rate:.1%}", f"Serious or fatal · {n_serious:,} collisions", "#16181D"),
        (f"{year_min}–{year_max}", "Coverage · Great Britain", "#16181D"),
        (f"{n_cleaned:,}", "Cleaned records used for modelling", "#4F5663"),
    ]
    cells = ""
    for i, (value, label, colour) in enumerate(stats):
        border = "border-left:1px solid #E1E5EA;" if i > 0 else ""
        cells += (
            f'<div class="rc-stat-cell" style="flex:1;padding:0 20px;{border}">'
            f'<div style="font-size:32px;font-weight:600;color:{colour}">{value}</div>'
            f'<div style="font-size:14px;color:#4F5663;margin-top:2px">{label}</div></div>'
        )
    st.markdown(
        '<div class="rc-stat-row" style="border-top:1px solid #E1E5EA;border-bottom:1px solid #E1E5EA;'
        f'padding:20px 0;display:flex;margin-top:8px">{cells}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size:13px;color:#687080;margin-top:14px">These figures describe police-reported '
        'collisions. Missing or unknown values may partly reflect reporting practices.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1px;background:#E1E5EA;margin:24px 0 32px"></div>', unsafe_allow_html=True)

    # ------------------------------------------------
    # Stage 9b: How often is the outcome serious or fatal?
    # ------------------------------------------------

    n_slight = n_raw - n_serious
    slight_rate = 1 - serious_rate
    one_in = round(1 / serious_rate) if serious_rate > 0 else 0
    takeaway = f"About 1 in {one_in} recorded collisions had a serious or fatal outcome."
    support = (
        f"This {slight_rate:.0%}/{serious_rate:.0%} imbalance is why the model is class-weighted during "
        "training and its output is calibrated afterwards."
    )
    stacked_bar_html = (
        '<div style="display:flex;height:40px;border-radius:6px;overflow:hidden">'
        f'<div style="width:{slight_rate * 100:.2f}%;background:#E9ECF0;color:#16181D;display:flex;'
        f'align-items:center;padding-left:12px;font-size:14px;font-weight:600">Slight</div>'
        f'<div style="width:{serious_rate * 100:.2f}%;background:#3A55D0;color:#FFFFFF;display:flex;'
        f'align-items:center;justify-content:flex-end;padding-right:12px;font-size:14px;font-weight:600">'
        f'{"Serious or fatal" if serious_rate > 0.12 else ""}</div></div>'
        '<div style="display:flex;margin-top:8px;font-size:13px;color:#4F5663">'
        f'<div style="width:{slight_rate * 100:.2f}%"><b style="color:#16181D">{slight_rate:.1%}</b> · {n_slight:,}</div>'
        f'<div style="width:{serious_rate * 100:.2f}%"><b style="color:#16181D">{serious_rate:.1%}</b> · {n_serious:,}</div>'
        '</div>'
    )
    left, right = st.columns([0.3, 0.7], gap="large")
    with left:
        st.markdown("#### How often is the outcome serious or fatal?")
        st.markdown(f'<div style="font-size:17px;color:#16181D">{takeaway}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:14px;color:#4F5663;margin-top:8px">{support}</div>', unsafe_allow_html=True)
        show_table_1 = st.toggle("View data table", key="table_toggle_severity")
    with right:
        st.markdown(stacked_bar_html, unsafe_allow_html=True)
        st.caption(f"{serious_rate:.1%} of recorded collisions ({n_serious:,} of {n_raw:,}) were serious or fatal.")
    if show_table_1:
        st.dataframe(
            pd.DataFrame({"Outcome": ["Slight", "Serious or fatal"], "Collisions": [n_slight, n_serious],
                          "Share": [f"{slight_rate:.1%}", f"{serious_rate:.1%}"]}),
            use_container_width=True, hide_index=True,
        )
    st.markdown('<div style="height:1px;background:#E1E5EA;margin:8px 0 32px"></div>', unsafe_allow_html=True)

    # ------------------------------------------------
    # Stage 9c: Severity rate by speed limit
    # ------------------------------------------------

    rate_by_speed = df[df["speed_limit"] > 0].groupby("speed_limit")["target"].agg(["mean", "count"]).sort_index()
    peak_speed = rate_by_speed["mean"].idxmax()
    peak_rate = rate_by_speed["mean"].max()
    top_speed = rate_by_speed.index.max()
    top_rate = rate_by_speed.loc[top_speed, "mean"]
    takeaway = f"The recorded rate rises with the speed limit, peaking at {peak_rate:.1%} on {peak_speed} mph roads."
    if top_speed != peak_speed:
        support = (
            f"{top_speed} mph roads, often motorways and dual carriageways, sit lower at {top_rate:.1%}. "
            "This is an association in recorded collisions, not evidence that a speed limit alone changes outcomes."
        )
    else:
        support = "This is an association in recorded collisions, not evidence that a speed limit alone changes outcomes."
    categories = [f"{s} mph" for s in rate_by_speed.index]
    fig = data_bar_fig(categories, rate_by_speed["mean"].values, rate_by_speed["count"].values, horizontal=False,
                        highlight=f"{peak_speed} mph", baseline=serious_rate, h=260)
    table_df = pd.DataFrame({"Speed limit": categories, "Collisions": rate_by_speed["count"].values,
                              "Serious or fatal": [f"{v:.1%}" for v in rate_by_speed["mean"].values]})
    story_row("Severity rate by speed limit", takeaway, support, fig, table_df, "table_toggle_speed",
              "Serious-or-fatal rate by recorded speed limit, with the dataset-wide rate as a dashed baseline.")

    # ------------------------------------------------
    # Stage 9d: Severity rate by road type
    # ------------------------------------------------

    ROAD_TYPE_NAMES = {1: "Roundabout", 2: "One-way street", 3: "Dual carriageway",
                        6: "Single carriageway", 7: "Slip road", 9: "Information unavailable"}
    rate_by_road = df.groupby("road_type")["target"].agg(["mean", "count"])
    rate_by_road.index = [ROAD_TYPE_NAMES[c] for c in rate_by_road.index]
    # sort by rate, but keep "Information unavailable" last regardless of where it would sort —
    # it marks incomplete recording, not a genuinely safer road type
    unavailable_label = "Information unavailable"
    rest = rate_by_road.drop(index=unavailable_label, errors="ignore").sort_values("mean", ascending=False)
    if unavailable_label in rate_by_road.index:
        rate_by_road = pd.concat([rest, rate_by_road.loc[[unavailable_label]]])
    else:
        rate_by_road = rest
    top_label = rate_by_road["mean"].idxmax()
    top_rate = rate_by_road["mean"].max()
    # road-type labels are all plural-friendly with a trailing "s" (carriageways, roundabouts,
    # roads, streets) — "Information unavailable" is never realistically the top (highest) rate,
    # since an unrecorded road type consistently shows the lowest rate in this dataset
    takeaway = f"{top_label}s have the highest recorded rate, at {top_rate:.1%}."
    if unavailable_label in rate_by_road.index and rate_by_road.loc[unavailable_label, "mean"] == rate_by_road["mean"].min():
        unavail_rate = rate_by_road.loc[unavailable_label, "mean"]
        support = (
            f"Collisions with no recorded road type have the lowest rate ({unavail_rate:.1%}). That more "
            "likely reflects less detailed reporting than safer roads."
        )
    else:
        support = "Values the model treats as unrecorded are drawn hatched, rather than sorted as a genuine category."
    fig = data_bar_fig(list(rate_by_road.index), rate_by_road["mean"].values, rate_by_road["count"].values,
                        horizontal=True, highlight=top_label, hatch=unavailable_label, baseline=serious_rate,
                        h=220, axis_max=max(0.40, rate_by_road["mean"].max() + 0.05))
    table_df = pd.DataFrame({"Road type": rate_by_road.index, "Collisions": rate_by_road["count"].values,
                              "Serious or fatal": [f"{v:.1%}" for v in rate_by_road["mean"].values]})
    story_row("Severity rate by road type", takeaway, support, fig, table_df, "table_toggle_road",
              f"Dataset baseline {serious_rate:.1%} · hatched = not recorded · axis 0–"
              f"{max(0.40, rate_by_road['mean'].max() + 0.05):.0%}")

    # ------------------------------------------------
    # Stage 9e: Urban and rural areas
    # ------------------------------------------------

    urban_rural_df = df[df["urban_or_rural_area"].isin([1, 2])]
    rate_by_area = urban_rural_df.groupby("urban_or_rural_area")["target"].agg(["mean", "count"])
    rate_by_area.index = ["Urban" if c == 1 else "Rural" for c in rate_by_area.index]
    rate_by_area = rate_by_area.sort_values("mean", ascending=False)
    higher_label = rate_by_area["mean"].idxmax()
    lower_label = rate_by_area["mean"].idxmin()
    higher_rate, lower_rate = rate_by_area["mean"].max(), rate_by_area["mean"].min()
    takeaway = f"{higher_label} collisions were recorded as serious or fatal at {higher_rate:.1%}, compared with {lower_rate:.1%} in {lower_label.lower()} areas."
    minority_label = rate_by_area["count"].idxmin()
    minority_share = rate_by_area.loc[minority_label, "count"] / rate_by_area["count"].sum()
    support = f"{minority_label} areas account for {minority_share:.0%} of recorded collisions."
    fig = data_bar_fig(list(rate_by_area.index), rate_by_area["mean"].values, rate_by_area["count"].values,
                        horizontal=True, highlight=higher_label, baseline=serious_rate, h=140)
    table_df = pd.DataFrame({"Area type": rate_by_area.index, "Collisions": rate_by_area["count"].values,
                              "Serious or fatal": [f"{v:.1%}" for v in rate_by_area["mean"].values]})
    story_row("Urban and rural areas", takeaway, support, fig, table_df, "table_toggle_area",
              "Serious-or-fatal rate by recorded area type.")

    # ------------------------------------------------
    # Stage 9f: Lighting conditions
    # ------------------------------------------------

    LIGHT_NAMES = {1: "Daylight", 4: "Dark – street lights lit", 5: "Dark – street lights unlit",
                   6: "Dark – no street lighting", 7: "Dark – lighting not recorded"}
    rate_by_light = df[df["light_conditions"].isin(LIGHT_NAMES.keys())].groupby("light_conditions")["target"].agg(["mean", "count"])
    rate_by_light.index = [LIGHT_NAMES[c] for c in rate_by_light.index]
    not_recorded_label = "Dark – lighting not recorded"
    rest = rate_by_light.drop(index=not_recorded_label, errors="ignore").sort_values("mean", ascending=False)
    if not_recorded_label in rate_by_light.index:
        rate_by_light = pd.concat([rest, rate_by_light.loc[[not_recorded_label]]])
    else:
        rate_by_light = rest
    top_light = rate_by_light["mean"].idxmax()
    top_light_rate = rate_by_light["mean"].max()
    takeaway = f"{top_light} has the highest recorded rate, at {top_light_rate:.1%}."
    support = (
        "This likely overlaps with road type, speed and area (rural, unlit roads tend to allow higher "
        "speeds) rather than being an independent effect."
    )
    fig = data_bar_fig(list(rate_by_light.index), rate_by_light["mean"].values, rate_by_light["count"].values,
                        horizontal=True, highlight=top_light, hatch=not_recorded_label, baseline=serious_rate, h=200)
    table_df = pd.DataFrame({"Lighting": rate_by_light.index, "Collisions": rate_by_light["count"].values,
                              "Serious or fatal": [f"{v:.1%}" for v in rate_by_light["mean"].values]})
    story_row("Lighting conditions", takeaway, support, fig, table_df, "table_toggle_light",
              "Serious-or-fatal rate by recorded lighting condition; hatched = lighting detail not recorded.")

    # ------------------------------------------------
    # Stage 9g: Footer
    # ------------------------------------------------

    footer_cols = st.columns([0.7, 0.3])
    with footer_cols[0]:
        st.markdown(
            '<div style="font-size:13px;color:#687080;padding-top:6px">Source: Department for Transport, '
            'STATS19 road collision data, Great Britain, 2020–2024.</div>',
            unsafe_allow_html=True,
        )
    with footer_cols[1]:
        if st.button("About the data", key="about_data_btn_data_tab"):
            about_data_dialog()

# ------------------------------------------------
# Stage 10: Model insights
# ------------------------------------------------

# load the SHAP values saved by the notebook for the 5,000-row test sample, if the file exists
@st.cache_resource
def load_shap_sample():
    path = BASE_DIR / "shap_sample.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as file:
        return pickle.load(file)

def feature_display_label(name):
    return FEATURE_LABELS.get(name, name)

def ordinal(n):
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"

# ranks every one of the model's 83 raw columns by mean(|SHAP|) across the evaluation sample —
# the global counterpart to grouped_factors()'s per-scenario view
def global_importance_order(sample):
    importance = np.abs(np.asarray(sample["values"], dtype=float)).mean(axis=0)
    order = np.argsort(-importance)
    return order, importance

# horizontal bars, top 3 in the accent colour (the ones named in the key finding), rest neutral
def insights_bar_fig(names, values, highlight_n, h):
    labels = [feature_display_label(n) for n in names]
    colours = [RC["accent"] if i < highlight_n else RC["neutral"] for i in range(len(names))]
    fig = go.Figure(go.Bar(
        orientation="h", y=labels, x=values, marker=dict(color=colours, cornerradius=2),
        text=[f"{v:.3f}" for v in values], textposition="outside",
        textfont=dict(size=13, color=RC["text"]), cliponaxis=False,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    rc_layout(fig, h)
    fig.update_layout(margin=dict(l=0, r=56, t=8, b=8), bargap=0.4)
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    return fig

# beeswarm of the given (already-ordered) features: one dot per evaluation collision, shaded by
# that collision's value for the feature, with alternating row bands and a centre zero-line
def insights_beeswarm_fig(sample, order, h):
    values = np.asarray(sample["values"], dtype=float)
    data = np.asarray(sample["data"], dtype=float)
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
        marker=dict(size=6, color=np.concatenate(colours), colorscale=[[0, "#C8CED7"], [1, RC["accent"]]]),
    ))
    for row in range(len(order)):
        if row % 2 == 1:
            fig.add_hrect(y0=row - 0.5, y1=row + 0.5, fillcolor="#F0F2F5", line_width=0, layer="below")
    fig.add_vline(x=0, line_color="#9AA2AE", line_width=1)
    rc_layout(fig, h)
    fig.update_layout(margin=dict(l=0, r=10, t=8, b=8))
    fig.update_xaxes(zeroline=False, tickfont=dict(size=12, color=RC["muted"]))
    fig.update_yaxes(autorange="reversed", zeroline=False, tickvals=list(range(len(order))), ticksuffix="  ",
        ticktext=[feature_display_label(sample["feature_names"][i]) for i in order])
    return fig

# the "Reading one feature: speed limit" note — the direction and the mph ranges it cites are
# both read from the evaluation sample, never hard-coded, since they could change with the model
def speed_limit_reading_note(sample, feature_names_eval):
    if "speed_limit" not in feature_names_eval:
        return "Speed limit is not among the model's current inputs."
    i = feature_names_eval.index("speed_limit")
    data = np.asarray(sample["data"], dtype=float)[:, i]
    values = np.asarray(sample["values"], dtype=float)[:, i]
    median = np.median(data)
    low_mask, high_mask = data <= median, data > median
    low_mean, high_mean = values[low_mask].mean(), values[high_mask].mean()
    low_range = f"{int(data[low_mask].min())}–{int(data[low_mask].max())} mph"
    high_range = f"{int(data[high_mask].min())}–{int(data[high_mask].max())} mph"
    if high_mean > low_mean:
        return (f"Dark dots (recorded limits of {high_range}) sit right of the centre line, so higher "
                f"recorded limits pushed estimates higher. Light dots ({low_range}) sit to the left and "
                "pushed estimates lower.")
    return (f"Dark dots (recorded limits of {high_range}) sit left of the centre line, so higher "
            f"recorded limits pushed estimates lower. Light dots ({low_range}) sit to the right and "
            "pushed estimates higher.")

with tab_shap:

    st.markdown("# How the model works")
    st.markdown(
        '<div style="font-size:15px;color:#4F5663;margin-top:-8px">Understand which recorded features '
        "have the greatest influence on the model's estimates.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:1px;background:#16181D;margin:20px 0 28px"></div>', unsafe_allow_html=True)

    shap_sample = load_shap_sample()

    if shap_sample is not None:
        order, importance = global_importance_order(shap_sample)
        feature_names_eval = list(shap_sample["feature_names"])
        TOP_N = 10
        top_order = order[:TOP_N]
        top_names = [feature_names_eval[i] for i in top_order]
        top_values = importance[top_order]

        # ---- key finding — every claim below is computed from this run's evaluation sample ----
        top3_groups = []
        for name in (feature_names_eval[i] for i in order):
            group = feature_display_label(name).split(":", 1)[0].strip()
            if group not in top3_groups:
                top3_groups.append(group)
            if len(top3_groups) == 3:
                break
        groups_joined = f"{top3_groups[0]}, {top3_groups[1]}, and {top3_groups[2].lower()}"

        CLAIM_N = 20
        weather_idxs = [i for i, name in enumerate(feature_names_eval) if name.startswith("weather_conditions_")]
        rank_of = {idx: r + 1 for r, idx in enumerate(order)}
        best_weather_rank = min((rank_of[i] for i in weather_idxs), default=None)
        if best_weather_rank is None or best_weather_rank > CLAIM_N:
            weather_note = f"No weather category appears among the {CLAIM_N} most influential features."
        else:
            weather_note = (
                f"The strongest weather category ranks {ordinal(best_weather_rank)} by influence — "
                f"still well behind {groups_joined}."
            )

        kf_left, kf_right = st.columns([0.22, 0.78])
        with kf_left:
            st.markdown(
                '<div style="font-size:13px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;'
                'color:#4F5663;margin-top:4px">Key finding</div>',
                unsafe_allow_html=True,
            )
        with kf_right:
            st.markdown(
                f'<div style="font-size:26px;font-weight:500;color:#16181D;line-height:1.3">'
                f'{groups_joined.capitalize()} influence the model more strongly than weather.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(f'<div style="font-size:14px;color:#4F5663;margin-top:8px">{weather_note}</div>',
                        unsafe_allow_html=True)
        st.markdown('<div style="height:1px;background:#E1E5EA;margin:28px 0"></div>', unsafe_allow_html=True)

        # ---- feature importance ----
        fi_left, fi_right = st.columns([0.3, 0.7], gap="large")
        with fi_left:
            st.markdown("#### Feature importance")
            st.markdown(
                f'<div style="font-size:17px;color:#16181D">{groups_joined.capitalize()} are the three '
                'strongest influences.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:14px;color:#4F5663;margin-top:8px">Each value is the average size '
                f"of a feature's influence on the model output across {len(shap_sample['values']):,} "
                'evaluation collisions, regardless of direction.</div>',
                unsafe_allow_html=True,
            )
            show_all = st.toggle(f"Show all {len(feature_names_eval)} features", key="show_all_features")
        with fi_right:
            if show_all:
                full_names = [feature_names_eval[i] for i in order]
                st.plotly_chart(insights_bar_fig(full_names, importance[order], 3, 18 * len(full_names) + 40),
                                use_container_width=True, config=CHART_CONFIG)
            else:
                st.plotly_chart(insights_bar_fig(top_names, top_values, 3, 34 * TOP_N + 40),
                                use_container_width=True, config=CHART_CONFIG)
            st.markdown(
                '<div style="display:flex;gap:20px;font-size:13px;color:#4F5663;margin-top:4px">'
                '<span><span style="display:inline-block;width:10px;height:10px;border-radius:2px;'
                'background:#3A55D0;margin-right:6px"></span>Named in the key finding</span>'
                '<span>Average influence, mean |SHAP| in log-odds</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:1px;background:#E1E5EA;margin:28px 0"></div>', unsafe_allow_html=True)

        # ---- direction of effect ----
        de_left, de_right = st.columns([0.3, 0.7], gap="large")
        with de_left:
            st.markdown("#### Direction of effect")
            st.markdown(
                '<div style="font-size:17px;color:#16181D">Importance says how much a feature matters. '
                'This chart shows which way it pushes.</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div style="font-size:14px;color:#4F5663;margin-top:8px">Each dot is one evaluation '
                "collision. Its horizontal position is how far that feature moved the estimate; its "
                "shade is the feature's value in that collision.</div>",
                unsafe_allow_html=True,
            )
            speed_note = speed_limit_reading_note(shap_sample, feature_names_eval)
            st.markdown(
                '<div style="border:1px solid #E1E5EA;border-radius:8px;padding:14px 16px;margin-top:16px">'
                '<div style="font-size:14px;font-weight:600;color:#16181D">Reading one feature: speed limit</div>'
                f'<div style="font-size:13px;color:#4F5663;margin-top:6px">{speed_note}</div></div>',
                unsafe_allow_html=True,
            )
        with de_right:
            st.markdown(
                '<div style="display:flex;justify-content:space-between;font-size:14px;color:#4F5663;'
                'margin-bottom:4px"><span>← Pushes estimate lower</span>'
                '<span>Pushes estimate higher →</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(insights_beeswarm_fig(shap_sample, top_order, 34 * TOP_N + 40),
                            use_container_width=True, config=CHART_CONFIG)
            st.markdown(
                '<div style="display:flex;align-items:center;gap:8px;font-size:13px;color:#4F5663">'
                '<span>Low feature value</span>'
                '<span style="display:inline-block;width:56px;height:8px;border-radius:4px;'
                'background:linear-gradient(to right,#C8CED7,#3A55D0)"></span>'
                '<span>High feature value</span></div>',
                unsafe_allow_html=True,
            )

        with st.expander("How to read these charts"):
            st.markdown(
                '<div style="font-size:14px;color:#4F5663">SHAP shows how recorded features influenced '
                "this model's output across the evaluation sample. It explains model behaviour, not "
                "real-world causation. Strong effects for missing or unknown categories may indicate "
                "patterns in data collection rather than genuine safety effects.</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("The SHAP sample file was not found — showing static reference charts instead.")
        st.image(str(BASE_DIR.parent / "graphs" / "shap_feature_importance.png"))
        st.image(str(BASE_DIR.parent / "graphs" / "shap_dot_plot.png"))

    st.markdown('<div style="height:1px;background:#E1E5EA;margin:28px 0"></div>', unsafe_allow_html=True)

    # ---- model information ----
    st.markdown("## Model information")
    st.markdown(
        '<div style="font-size:15px;color:#4F5663;margin-top:-8px">Figures come from the held-out '
        'evaluation set and are loaded from the model bundle, so they stay in step with the deployed '
        'model.</div>',
        unsafe_allow_html=True,
    )

    def info_row(left_label, left_value, right_label, right_value):
        return (
            '<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:14px 0;'
            'border-top:1px solid #E1E5EA">'
            f'<div><div style="font-size:14px;color:#4F5663">{left_label}</div>'
            f'<div style="font-size:15px;font-weight:600;color:#16181D;margin-top:2px">{left_value}</div></div>'
            f'<div><div style="font-size:14px;color:#4F5663">{right_label}</div>'
            f'<div style="font-size:15px;font-weight:600;color:#16181D;margin-top:2px">{right_value}</div></div>'
            '</div>'
        )

    if METRICS:
        rows_html = (
            info_row("Model", "Tuned, class-weighted XGBoost", "Probability calibration", "Isotonic regression")
            + info_row("Evaluation set", f"{METRICS.get('eval_set_size', 0):,} held-out collisions",
                        "Current alert threshold", f"{threshold:.1%}")
            + info_row("ROC-AUC", f"{METRICS.get('roc_auc', float('nan')):.3f}",
                        "PR-AUC, before calibration", f"{METRICS.get('pr_auc_uncalibrated', float('nan')):.3f}")
            + info_row("Brier score", f"{METRICS.get('brier_score', float('nan')):.3f}",
                        "Average calibration gap",
                        f"Under {METRICS.get('calibration_gap_pct_points', float('nan')):.1f} percentage points")
        )
        st.markdown(rows_html + '<div style="border-bottom:1px solid #E1E5EA"></div>', unsafe_allow_html=True)

        st.markdown(
            '<div style="margin-top:24px"><div style="font-size:15px;font-weight:600;color:#16181D">'
            'What calibrated means</div>'
            '<div style="font-size:14px;color:#4F5663;margin-top:4px">When the model predicts '
            'approximately 30%, about 30% of comparable cases in the evaluation data had a serious or '
            f'fatal outcome. {METRICS.get("calibration_method", "")}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="margin-top:20px"><div style="font-size:15px;font-weight:600;color:#16181D">'
            f'Why the alert threshold is {threshold:.1%}</div>'
            f'<div style="font-size:14px;color:#4F5663;margin-top:4px">{METRICS.get("threshold_rationale", "")}</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Evaluation metadata was not found in the model bundle.")
