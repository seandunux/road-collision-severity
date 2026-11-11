# Handoff: RoadCast — light redesign of the Streamlit app

## Overview
A light-theme redesign of `app/app.py`. It replaces the dark "Nocturne" handoff in `../design_handoff_streamlit_redesign/` (now superseded). The model logic, `grouped_factors()`, the staleness check, the session-state keys and the `fkey()` reset pattern all stay. What changes:

- **Product name and tabs.** The app becomes **RoadCast**, with tabs **Predict · Data · Model insights**. The old tabs were "Risk Predictor · Dataset Overview · SHAP Explorer".
- **Light theme.** White page, near-black text, one blue accent (`#3A55D0`).
- **No "risk" wording.** HIGH RISK / LOWER RISK becomes **"Above / Below alert threshold"**, and the **dataset baseline** always appears beside the threshold.
- **Comparison scale.** A new 0–60% scale shows the estimate, threshold and baseline as three different marks.
- **Plain-language explanation first.** "Factors increasing / reducing this estimate" lead; the SHAP waterfall moves into an expander.
- **Data tab as a narrative.** One question per row: the takeaway on the left, the chart on the right, and a data table toggle.
- **Model insights tab.** Opens with a one-sentence finding. Feature importance shows the top 10 with "Show all", and the model metrics sit in a compact list.

## About the design files
- `design/*.dc.html` is a **visual reference built in HTML. Do not ship it.** Recreate it with Streamlit widgets, `.streamlit/config.toml`, one injected CSS block, small HTML snippets via `st.markdown(..., unsafe_allow_html=True)`, and Plotly.
- **To view the reference**, serve this folder (`python -m http.server` inside `design/`) and open the `.dc.html` files. They need `support.js` and `roadcast/roadcast.css` beside them.
- `roadcast/roadcast.css` is the **source of truth for every token and measurement** below.
- `screenshots/` holds one PNG per screen and state, at 1440px desktop and 390px mobile width.

## ⚠️ Values that must stay dynamic
Every probability, SHAP contribution, difference and strength label in the mockups is **an example**. Never hard-code them.

| Value in the mockup | Source in the real app |
|---|---|
| Alert threshold 22.5% | `save_model_data["threshold"]` (already `threshold`) |
| Dataset baseline 23.3% | `METRICS["dataset_prevalence"]`, or `df["target"].mean()` |
| ROC-AUC, PR-AUC, Brier score, calibration gap, evaluation set size | `METRICS[...]` (already used in "About this model") |
| Data-tab rates and counts | Compute from `load_dataset()`, cached. Mock values came from the full 503,475-row `dataset.csv`. |
| Feature importance and beeswarm | `load_shap_sample()` |
| Factor rows | `grouped_factors(explanation)` |

The threshold currently sits **just below** the baseline. All comparison logic must therefore handle an estimate that is above one and below the other: compute each difference independently, and never infer one from the other.

## Screens

| # | Screenshot | What it shows |
|---|---|---|
| 01 | predict-empty | Before the first calculation. Compact empty card, plus the scale with only the threshold and baseline markers. |
| 02 | predict-calculating | Primary button disabled with "Calculating…". Skeleton in the result card. |
| 03 | predict-above-threshold | Full result and factor rows. |
| 04 | predict-below-threshold-technical | Below-threshold result with "View technical explanation" expanded (the waterfall). |
| 05 | predict-outdated | An input changed after calculating: neutral banner, result at 50% opacity, "Changed" badge on the field. |
| 06 | predict-missing-information | Two inputs set to "Information unavailable": amber note, dashed fields, "Unrecorded" badges. |
| 07 | predict-error | Model error: red alert, "Try again", "Technical details" expander. |
| 08–09 | predict-mobile-* | Single column, primary action pinned, result after scrolling. |
| 10 | data-desktop | Data tab narrative. |
| 11 | model-insights-desktop | Model insights tab. |
| 12 | data-mobile | Data tab at 390px; every chart becomes horizontal bars. |
| 13 | about-the-data-dialog | `st.dialog` content. |
| 14 | component-system | Every component with its states. |
| 15 | design-rationale | Why the design makes each choice. |

## Shared header (all tabs)
Replace `st.title(...)` and the "Model loaded successfully!" line with a compact header:

- **Wordmark:** "RoadCast", 19px/600, with a 24px near-black rounded square mark. Under it, at 14px `#4F5663`: *"Explore how recorded road and environmental conditions relate to collision severity."*
- **"About the model":** a quiet text button on the right, opening an `st.dialog` with the existing METRICS content. **Remove** the success message entirely; on a load failure, show `st.error` instead.
- **Tabs:** `st.tabs(["Predict", "Data", "Model insights"])`, at 15px.
  - Inactive tabs: `#4F5663`, weight 500.
  - Active tab: `#16181D`, **weight 600**, with a 2px `#3A55D0` underline. This must not rely on colour alone.
- **Page padding:** content max-width 1280px, 80px side padding on desktop, 16px on mobile.

## Predict tab
**Layout:** `left, right = st.columns([0.37, 0.63], gap="large")`. The mock uses a 464px input column, a 64px gap and a 1280px content width.

### Inputs (left), no bordered card
- **Heading:** `### Configure a scenario` (30px/600), with the subtitle "Choose the recorded conditions for a hypothetical collision." at 15px `#4F5663`.
- **Groups:** three groups, each headed by a 13px/600 uppercase, letter-spaced label in `#4F5663`, separated by a 1px `#E1E5EA` rule. Each group uses two columns (`st.columns(2)`):
  - **Road:** Speed limit | Area type; Road type | Junction control.
  - **Environment:** Lighting | Weather; Road surface | (empty).
  - **Time:** Day of week | Month (both selectboxes, month *names*); then Time of day as a full-width `select_slider` with options "00:00"…"23:00".
- **Road surface is a new input.** The app currently fixes it to Dry. The model already has `road_surface_conditions_*` (codes 1 Dry, 2 Wet or damp, 3 Snow, 4 Frost or ice, 5 Flood, 9 Information unavailable). Expose it, and remove it from "Values the model assumes".
- **Option labels** use sentence case: "Single carriageway", "Dark – no street lighting", "Dark – street lights lit", "Wet or damp". Keep the existing code maps and update only the labels.
- **"Information unavailable":** always the **last** option. When it's selected:
  - show an "Unrecorded" badge beside the label (13px, `#EEF0F3` background, `#4F5663` text)
  - show the hint *"Not recorded. May lower the estimate for reasons unrelated to the actual conditions."* (13px, `#687080`)
  - optionally give the field a dashed border via CSS
- **Help:** keep the existing `help=` tooltips, shortened to one or two sentences (see `design/RoadCast Predict.dc.html`, `HELP` object).
- **Expanders:** two in a row, below the groups.
  - "More road details" holds Primary road class, Trunk road, Special conditions and Pedestrian crossing.
  - "Values the model assumes" holds junction layout, second road and carriageway hazards.
  - Put a one-line summary of the current values in each expander's label, e.g. *"More road details — A road · Trunk road unrecorded · No special conditions · No crossing nearby"*.
- **Buttons:** one row, `st.columns([0.62, 0.38])`.
  - Primary: **"Calculate estimate"** (`type="primary"`), filled `#3A55D0` with white text, 44px tall. Its label becomes **"Calculate again"** when the result is outdated.
  - Secondary: **"Reset scenario"**, white with a 1px `#8A919D` border.
- **Changed fields:** after a calculation, any field whose value differs from the stored scenario gets a "Changed" badge (`#EEF1FC` background, `#2C43AE` text). The existing staleness comparison already has the data.

### Result (right)
`st.container(border=True)`: 1px `#E1E5EA` border, 12px radius, 28px 32px padding. This is the **only bordered surface** on the page.

**Empty** (no result in session state):
- `## Your estimate will appear here` (21px/600).
- Body: *"Configure the recorded road, environmental, and time conditions, then calculate the estimate."*
- The comparison scale showing only the threshold and baseline markers.
- Context line: *"Based on police-reported STATS19 collisions from Great Britain, 2020–2024."* (13px `#687080`).
- Do **not** show a `—%` placeholder.

**Calculating:** wrap the predict call in `st.spinner("Calculating estimate…")`. Optionally render a skeleton (grey `#EDEFF2` blocks) in the card.

**Result**, top to bottom:
1. **Kicker:** "Estimated probability of a serious or fatal outcome" (15px/500, `#4F5663`).
2. **Probability:** `f"{p:.1%}"` at **72px/600**, letter-spacing −0.035em, tabular figures, `#16181D`, with **no colour change and no glow**. Mobile: 52px.
3. **Status chip** beside the number:
   - Above: `#EEF1FC` background, `#2C43AE` text, 1px `#B7C1EE` inset ring, filled ▲, "Above alert threshold".
   - Below: `#F1F3F5` background, `#3D434E` text, 1px `#C8CED7` ring, hollow ▽, "Below alert threshold".
   - Pill shape, 32px tall, 14px/600.
4. **Comparison pair:** two columns between 1px rules.
   - Left: a solid-line glyph, "Alert threshold", `{t:.1%}` at 20px/600, and *"{|p−t|×100:.1f} points above/below"*.
   - Right: a dashed-line glyph, "Dataset baseline", `{b:.1%}`, and the same kind of difference.
5. **Comparison scale:** see the snippet below.
6. *"The alert threshold reflects the model's selected precision–recall trade-off. It is not a boundary between safe and dangerous conditions."* (14px `#4F5663`).
7. **Footer row**, above a 1px rule: the context line on the left, and an "About the data" button on the right that opens an `st.dialog` (content in screenshot 13).

**Outdated:** the staleness check is already in place.
- Above the result, show a neutral banner: `#F7F8FA` background, 1px `#C8CED7` border, refresh icon. Title **"This estimate is out of date"**; body *"{Field} changed from {old} to {new}. Calculate again to update it."*; and a small primary "Calculate again" button.
- Add an "Outdated" badge beside the kicker.
- Render the card body and the explanation at `opacity: 0.5`.

**Missing information:** if any input is "Information unavailable", put an amber note at the top of the card.
- Style: `#FFF6E5` background, 1px `#EBC27A` border, title `#7A4600`.
- Title: **"{n} inputs are set to Information unavailable"**.
- Body: name the fields, then *"In the training data, unrecorded values often accompany less detailed reports, which can lower the estimate for reasons unrelated to the road or weather."*

**Error:** catch exceptions around predict, calibration and SHAP.
- Show a red alert: `#FEF3F2` background, 1px `#F2B8B2` border, title `#B42318` **"The estimate couldn't be calculated"**, body *"The model didn't return a result. Your scenario hasn't changed, so you can try again. If this keeps happening, reload the page."*
- Below it, a secondary "Try again" button, and an `st.expander("Technical details")` holding the exception text.
- Use `st.error` only for the real error, never for a below-threshold result.

### Explanation, below the card (not boxed)
- `## What influenced this estimate`, then *"These are model influences, not proven causes. Missing or unknown values may reflect recording practices."*
- **Two columns:**
  - "↑ Factors increasing this estimate": rows from `grouped_factors()` with value > 0.
  - "↓ Factors reducing this estimate": rows with value < 0.
  - Each column heading is 15px/600 with a 1px rule beneath it.
- **Each row** is two columns, 1fr and 124px, with a 1px rule between rows:
  - Left: the feature name (15px/500) above the value used (14px `#4F5663`). Split the existing labels ("Speed Limit: 60 mph") on the colon. Add an "Unrecorded" badge if the value is "Information unavailable".
  - Right: an 8px track (`#E9ECF0`) filled to `|v| / max_abs_across_app` (use ~0.45), in `#3A55D0` for increasing and `#7A818D` for reducing. Below the track, a strength word: **Strong** (|v| ≥ 0.15), **Moderate** (≥ 0.07) or **Slight**.
- **Technical explanation:** `st.expander("View technical explanation")` holds the existing `waterfall_fig`, restyled (see Plotly). Precede it with one paragraph: *the raw output is in log-odds; the percentage is produced by a separate calibration step, so the scales don't map one-to-one.*

## Data tab
- **Header:** `# Understand the data` and *"Explore the police-reported collision records used to train and evaluate the model."*
- **Statistics row:** replaces the three `st.metric` cards. One HTML row of four cells, separated by 1px vertical rules, between top and bottom rules:
  - **503,475** Raw collision records
  - **23.3%** Serious or fatal · {count} collisions
  - **2020–2024** Coverage · Great Britain
  - **503,022** Cleaned records used for modelling (value in `#4F5663`)
  - Values 32px/600, labels 14px `#4F5663`.
- Then: *"These figures describe police-reported collisions. Missing or unknown values may partly reflect reporting practices."*
- **Story rows:** separated by 1px rules, each `st.columns([0.3, 0.7], gap="large")`. Left column: an `##` question, a takeaway sentence at 17px, a supporting line at 14px `#4F5663`, and a `st.toggle("View data table")` that shows `st.dataframe`. Right column: the chart.
  1. **How often is the outcome serious or fatal?** A single 100% stacked horizontal bar: Slight in `#E9ECF0` with dark text, Serious or fatal in `#3A55D0` with white text. Show the % and count below each segment.
  2. **Severity rate by speed limit.** Vertical bars. Highlight the maximum in `#3A55D0`; others `#8A919D`. Add a dashed baseline line labelled "Baseline 23.3%".
  3. **Severity rate by road type.** Horizontal bars, sorted descending, with a baseline line. Draw "Information unavailable" **hatched** (Plotly `marker.pattern.shape="/"`, white fill, `#8A919D` lines) and keep it last.
  4. **Urban and rural areas.** Two horizontal bars, Rural highlighted.
  5. **Lighting conditions.** A new chart from `light_conditions`. Horizontal bars; "Dark – lighting not recorded" hatched.
- **Takeaways** must be generated from the computed rates, never hard-coded. Wording must not imply causation: "recorded rate", "association".
- **Footer:** source line plus "About the data".

## Model insights tab
- **Header:** `# How the model works` and *"Understand which recorded features have the greatest influence on the model's estimates."*
- **Key finding:** a 1px near-black top rule, a small "KEY FINDING" label on the left, and 26px/500 text: *"Road structure, speed, and rural location influence the model more strongly than weather."* Then *"No weather category appears among the 20 most influential features."* Check that claim against the SHAP sample at runtime, or keep it static and re-verify when the model changes.
- **Feature importance:** top **10**, horizontal, 16px-tall bars.
  - Top 3 in `#3A55D0`, the rest `#8A919D`; values at 3 decimal places to the right of each bar.
  - Footer: legend "Named in the key finding" and "Average influence, mean |SHAP| in log-odds".
  - `st.button("Show all 83 features")`, or a toggle, renders the full ranking.
- **Direction of effect:** beeswarm of the top 10.
  - White background, a `#9AA2AE` zero line, and `#F0F2F5` row separators.
  - Headers above the plot: **"← Pushes estimate lower"** and **"Pushes estimate higher →"**.
  - Colour scale from `#C8CED7` (low feature value) to `#3A55D0` (high), which varies in lightness as well as hue.
  - Legend: "Low feature value ▭ High feature value".
  - Beside the chart, a bordered note **"Reading one feature: speed limit"** (copy in the reference).
- **"How to read these charts":** an `st.expander` with the SHAP caveat copy.
- **Model information:** two-column `<dl>`, built from METRICS and `threshold`: Model · Probability calibration · Evaluation set · Current alert threshold · ROC-AUC · PR-AUC before calibration · Brier score · Average calibration gap. Then two short sections: "What calibrated means" and "Why the alert threshold is {t:.1%}" (use `METRICS["threshold_rationale"]`).

## Streamlit implementation

### `.streamlit/config.toml` (replaces the dark theme)
```toml
[theme]
base = "light"
primaryColor = "#3A55D0"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F7F8FA"
textColor = "#16181D"
borderColor = "#E1E5EA"
baseRadius = "8px"
font = "Inter, sans-serif"
headingFontWeights = 600
```

### Injected CSS
This replaces the existing `<style>` block near the top of `app.py`.
```css
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="st-"], .stMarkdown { font-family: 'Inter', sans-serif; }
.block-container { max-width: 1280px; padding: 2rem 5rem 5rem; }
@media (max-width: 640px) { .block-container { padding: 1.25rem 1rem 6rem; } }
h1 { font-size: 30px !important; font-weight: 600 !important; letter-spacing: -0.02em; }
h2 { font-size: 21px !important; font-weight: 600 !important; }
/* tabs: weight + underline, not colour alone */
.stTabs [data-baseweb="tab"] { font-size: 15px; font-weight: 500; color: #4F5663; min-height: 44px; }
.stTabs [aria-selected="true"] { color: #16181D; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background: #3A55D0; height: 2px; }
/* controls: 44px, 3:1 edge */
[data-baseweb="select"] > div { min-height: 44px; border-color: #8A919D; }
.stButton > button { min-height: 44px; font-weight: 600; }
.stButton > button[kind="secondary"] { border-color: #8A919D; }
/* group heading */
.rc-group-h { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: #4F5663;
              border-top: 1px solid #E1E5EA; padding-top: 20px; margin-top: 4px; }
/* mobile: pin the primary action */
@media (max-width: 640px) {
  .st-key-predict_actions { position: sticky; bottom: 0; z-index: 10; background: rgba(255,255,255,.96);
                            border-top: 1px solid #E1E5EA; padding: 12px 0 16px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
```
Streamlit's `data-testid` attributes and class names change between versions, so confirm each selector against 1.50 in devtools. For the sticky bar, wrap the buttons in `st.container(key="predict_actions")`.

### Comparison scale (HTML via `st.markdown`)
```python
def scale_html(p, t, b, max_pct=60):
    pos = lambda v: f"{min(v * 100, max_pct) / max_pct * 100:.2f}%"
    est = (f'<div style="position:absolute;top:0;left:{pos(p)};transform:translateX(-50%);'
           f'font:600 14px/20px Inter;white-space:nowrap">This estimate {p:.1%}</div>') if p is not None else ""
    fill = f'<div style="position:absolute;inset:0 auto 0 0;width:{pos(p)};border-radius:5px;background:#B7C1EE"></div>' if p is not None else ""
    dot = (f'<div style="position:absolute;top:50%;left:{pos(p)};width:18px;height:18px;margin:-9px 0 0 -9px;'
           f'border-radius:50%;background:#3A55D0;border:3px solid #fff;box-shadow:0 0 0 1px #3A55D0"></div>') if p is not None else ""
    ticks = "".join(f"<span>{v}%</span>" for v in range(0, max_pct + 1, 10))
    return f'''
<figure style="margin:0" role="img" aria-label="Scale from 0 to {max_pct} percent. Estimate {'' if p is None else f'{p:.1%}'}; alert threshold {t:.1%}; dataset baseline {b:.1%}.">
 <div style="position:relative;padding-top:{34 if p is not None else 8}px">{est}
  <div style="position:relative;height:10px;border-radius:5px;background:#E9ECF0">{fill}
   <div style="position:absolute;top:-8px;height:26px;left:{pos(t)};margin-left:-1px;border-left:2px solid #16181D"></div>
   <div style="position:absolute;top:-8px;height:26px;left:{pos(b)};margin-left:-1px;border-left:2px dashed #4F5663"></div>{dot}
  </div></div>
 <div style="position:relative;height:44px;margin-top:6px;font:13px/18px Inter">
  <div style="position:absolute;left:{pos(t)};transform:translateX(-100%);padding-right:6px;text-align:right;white-space:nowrap">
   <b style="display:block">{t:.1%}</b>Alert threshold</div>
  <div style="position:absolute;left:{pos(b)};padding-left:6px;color:#4F5663;white-space:nowrap">
   <b style="display:block;color:#16181D">{b:.1%}</b>Dataset baseline</div>
 </div>
 <div style="display:flex;justify-content:space-between;font:13px/18px Inter;color:#687080">{ticks}</div>
</figure>'''
```
- **Label placement:** the threshold label is right-aligned to its line and the baseline label left-aligned to its own. When the threshold is *above* the baseline, swap the alignments.
- **Axis maximum:** if estimates can exceed 60%, raise `max_pct` to `max(60, ceil(p*10)*10 + 10)`.

### Status chip
```python
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
```

### Shared Plotly look (replaces `noc_layout`)
```python
RC = dict(text="#16181D", text2="#4F5663", muted="#687080", rule="#C8CED7",
          accent="#3A55D0", neutral="#8A919D", neg="#7A818D", track="#E9ECF0")
def rc_layout(fig, h):
    fig.update_layout(height=h, paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", size=14, color=RC["text"]),
        margin=dict(l=0, r=48, t=8, b=8), showlegend=False, bargap=0.35)
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=14, color=RC["text"]))
    return fig
# bars: marker_color per bar (accent for the takeaway's bar, neutral otherwise),
#       texttemplate="<b>%{x:.1%}</b>  %{customdata:,}", textposition="outside"
# baseline: fig.add_vline(x=b, line_dash="dash", line_width=1.5, line_color=RC["text2"],
#       annotation_text=f"Baseline {b:.1%}", annotation_position="top right")
# unrecorded: marker=dict(color="#FFFFFF", line=dict(color="#8A919D", width=1),
#       pattern=dict(shape="/", fgcolor="#C8CED7", size=6))
# st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
```
**Waterfall:** keep the existing logic. Positive contributions are `#3A55D0` and negative `#7A818D`. Add a vertical line at E[f(x)] in `#C8CED7`, with value labels at 13px/600 `#16181D`. Make labels readable, e.g. "Speed limit · 30" and "Rural area · no".

**Beeswarm:** use `colorscale=[[0,"#C8CED7"],[1,"#3A55D0"]]`, marker size 6, a zero line in `#9AA2AE`, and `plot_bgcolor` white.

## Design tokens

| Token | Value | Use / contrast on white |
|---|---|---|
| Page / secondary | `#FFFFFF` / `#F7F8FA` | |
| Primary text | `#16181D` | 17.8:1 |
| Secondary text | `#4F5663` | 7.4:1 (6.9:1 on `#F7F8FA`) |
| Muted text | `#687080` | 5.0:1. Don't use `#6F7785`: it measures 4.3:1 on `#F7F8FA`. |
| Border / strong border | `#E1E5EA` / `#C8CED7` | Dividers and decoration only |
| Control edge | `#8A919D` | 3.2:1; selects, secondary buttons, context bars |
| Accent | `#3A55D0` (hover `#2E45B0`, pressed `#263A94`) | 6.2:1; positive influence, estimate, highlighted bar |
| Accent weak / line | `#EEF1FC` / `#B7C1EE` | Chip background, scale fill |
| Negative influence | `#7A818D` | 3.9:1; lighter than the accent, so the two separate in greyscale |
| Warning | `#FFF6E5` background, `#EBC27A` line, `#7A4600` text | Unrecorded inputs only |
| Error | `#FEF3F2` background, `#F2B8B2` line, `#B42318` text | Real errors only |

- **Type:** Inter. Body 15/24, labels 14/20/500, captions 13/19 (minimum), headings 600, big number 72/600, statistics 32/600. Use tabular figures for every number.
- **Radius:** 8px for controls, 12px for the result card, 999px for chips.
- **Spacing:** 40–64px between major blocks; 16–18px between fields.
- **Shadows:** none, except dialogs and menus.

## Accessibility checklist
- **Status** is shown by word and shape, never by colour alone: ▲ filled vs ▽ hollow, and solid vs dashed reference lines.
- **Charts:** each has a one-sentence text alternative (`st.caption` under the chart) and a data-table toggle.
- **Targets and text:** touch targets are at least 44px; no text is under 13px.
- **Focus:** keep a visible keyboard focus ring (2px `#3A55D0`); don't remove outlines.
- **Motion:** respect `prefers-reduced-motion` (in the CSS above).

## Files
- `design/RoadCast Predict.dc.html`: desktop states P1–P7, mobile M1–M2.
- `design/RoadCast Data and Insights.dc.html`: Data, Model insights, Data on mobile, About the data dialog.
- `design/RoadCast System.dc.html`: component system and design rationale.
- `design/roadcast/roadcast.css`: all tokens and component measurements.
- `design/support.js`: needed only to open the reference in a browser.
- `screenshots/01…15`: one PNG per screen and state.
