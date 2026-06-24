"""
Where Should We Expand? -- Manager Dashboard
DPIIT Startup India data (dataset 15737), state-level expansion decision support.

Run locally:   streamlit run app.py
Deploy:        push this file + pipeline.py + india_states_map.b64 to a GitHub repo,
               deploy via Streamlit Community Cloud (share.streamlit.io), point it at app.py.
"""

import os

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import pipeline as pl

NAVY = "#1F3864"
ACCENT = "#C0143C"
WEIGHT_PRESETS = pl.WEIGHT_PRESETS

st.set_page_config(page_title="Where Should We Expand?", page_icon="\U0001F4CA", layout="wide")

# =============================================================================
# CACHED WRAPPERS -- the actual logic lives in pipeline.py (tested independently
# with plain Python; see the project notes). Caching here is keyed on inputs, so
# changing one sidebar control only recomputes what actually depends on it.
# =============================================================================

load_and_clean = st.cache_data(show_spinner="Reading and cleaning the dataset...")(pl.load_and_clean)
get_geojson = st.cache_data(pl.get_geojson)
compute_eligibility = st.cache_data(show_spinner="Computing state totals and the reliability gate...")(pl.compute_eligibility)
compute_growth = st.cache_data(show_spinner="Fitting growth trends (with size-bias correction + bootstrap)...")(pl.compute_growth)
compute_lq = st.cache_data(show_spinner="Computing sector concentration (Location Quotient)...")(pl.compute_lq)
compute_sector_growth = st.cache_data(show_spinner="Computing sector-specific growth...")(pl.compute_sector_growth)
compute_clusters = st.cache_data(show_spinner="Clustering states into ecosystem tiers...")(pl.compute_clusters)
compute_scores = st.cache_data(show_spinner="Scoring states...")(pl.compute_scores)
score_with_weights = st.cache_data(pl.score_with_weights)
monte_carlo_stability = st.cache_data(show_spinner="Testing stability across random weightings...")(pl.monte_carlo_stability)

# =============================================================================
# SIDEBAR -- the manager's controls
# =============================================================================

st.sidebar.title("\U0001F4CA Controls")
uploaded = st.sidebar.file_uploader("DPIIT dataset (.zip or .csv)", type=["zip", "csv"])

default_path = "15737-_Dataful.zip"
if uploaded is None and os.path.exists(default_path):
    with open(default_path, "rb") as f:
        file_bytes, filename = f.read(), default_path
elif uploaded is not None:
    file_bytes, filename = uploaded.getvalue(), uploaded.name
else:
    st.title("Where Should We Expand?")
    st.info("\U0001F446 Upload the DPIIT dataset (15737-_Dataful.zip, or the extracted CSV) in the sidebar to begin.")
    st.stop()

try:
    df, STABLE_YEARS, PROVISIONAL_YEARS, GROWTH_YEARS = load_and_clean(file_bytes, filename)
except Exception as e:
    st.error(f"Couldn't read this file: {e}")
    st.stop()

all_industries = sorted(df["industry"].unique())
all_states = sorted(df["state"].unique())
default_sector = "IT Services" if "IT Services" in all_industries else all_industries[0]
default_home = "Karnataka" if "Karnataka" in all_states else all_states[0]

target_sector = st.sidebar.selectbox("Target sector", all_industries,
                                      index=all_industries.index(default_sector))
home_state = st.sidebar.selectbox("Home state (excluded from picks)", all_states,
                                   index=all_states.index(default_home))
min_threshold = st.sidebar.slider("Minimum scale to trust a state's ratios", 50, 1000, 200, step=50,
                                   help="States below this total startup count are excluded from growth/fit "
                                        "rankings -- their ratios are too noisy on a tiny sample to trust.")
top_n = st.sidebar.slider("How many states to recommend", 3, 10, 5)

st.sidebar.markdown("**Weighting**")
weight_mode = st.sidebar.radio("Scenario", ["Balanced", "Growth-tilted", "Scale-tilted", "Fit-tilted", "Custom"],
                                label_visibility="collapsed")
if weight_mode == "Custom":
    c1 = st.sidebar.slider("Scale weight", 0.0, 1.0, 0.34)
    c2 = st.sidebar.slider("Growth weight", 0.0, 1.0, 0.33)
    c3 = st.sidebar.slider("Sector-fit weight", 0.0, 1.0, 0.33)
    total = max(c1 + c2 + c3, 1e-6)
    active_weights = {"scale_n": c1 / total, "growth_n": c2 / total, "fit_n": c3 / total}
else:
    active_weights = WEIGHT_PRESETS[weight_mode]

st.sidebar.caption(f"{len(PROVISIONAL_YEARS)} provisional year(s) excluded from growth/scoring. "
                    f"Growth measured {GROWTH_YEARS[0]}\u2013{GROWTH_YEARS[-1]}.")

# =============================================================================
# RUN THE PIPELINE
# =============================================================================

stable, state_totals, eligible_states = compute_eligibility(df, STABLE_YEARS, min_threshold)
growth_df = compute_growth(stable, state_totals, eligible_states, GROWTH_YEARS)
lq, state_share = compute_lq(stable)
sector_growth = compute_sector_growth(stable, eligible_states, GROWTH_YEARS, target_sector)
features = compute_clusters(state_totals, growth_df, state_share, eligible_states)

recent_years = STABLE_YEARS[-3:]
# `scores` holds the 4 fixed preset scenarios + their average -- used only as a robustness reference
# (Tab 5). The number actually shown as THE score everywhere else is `selected_score`, computed under
# whatever weighting the manager picked in the sidebar, so the sidebar controls actually do something.
scores, norm, score_inputs = compute_scores(features, growth_df, lq, sector_growth, state_totals, stable,
                                             recent_years, target_sector)
selected_score = score_with_weights(norm, active_weights).rename("score")

ranked = selected_score.sort_values(ascending=False)
display_scores = ranked.drop(index=home_state, errors="ignore")
top_states = display_scores.head(top_n).index.tolist()
top5_pct = monte_carlo_stability(norm, home_state, top_n)

geojson = get_geojson()

# =============================================================================
# TABS
# =============================================================================

st.title("\U0001F4CA Where Should We Expand?")
st.caption(f"DPIIT startup-recognition data, {STABLE_YEARS[0]}\u2013{STABLE_YEARS[-1]} \u00b7 "
           f"target sector: **{target_sector}** \u00b7 current HQ: **{home_state}** (excluded from picks)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["\U0001F3C6 Recommendation", "\U0001F50D Explore a State", "\u2696\uFE0F Compare States",
     "\U0001F3AF Opportunity vs. Saturation", "\U0001F52C Robustness & Notes"])

# ---------- TAB 1: RECOMMENDATION ----------
with tab1:
    if not top_states:
        st.warning("No states pass the current reliability threshold. Lower it in the sidebar.")
    else:
        winner = top_states[0]
        w_score = display_scores.loc[winner]
        w_stability = top5_pct.get(winner, 0.0)
        w_read = "growing" if growth_df.loc[winner, "trend_growth_pct"] > 0 else "flat or shrinking"

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Top pick", winner)
        c2.metric("Composite score", f"{w_score:.0f}/100")
        c3.metric("Stability", f"{w_stability:.0f}%", help="% of 500 random weight draws where this state stayed in the top list")
        c4.metric(f"{target_sector} LQ", f"{lq.loc[winner, target_sector]:.2f}x")

        st.markdown(
            f"**{winner}** is the strongest candidate for expanding into **{target_sector}** outside {home_state}, "
            f"with {int(state_totals.set_index('state').loc[winner,'total_startups']):,} startups recognized "
            f"({STABLE_YEARS[0]}\u2013{STABLE_YEARS[-1]}) and an ecosystem that is {w_read} "
            f"({growth_df.loc[winner,'trend_growth_pct']:+.0f}%/yr). It places in the top {top_n} under "
            f"{w_stability:.0f}% of random weight combinations tested, making it a comparatively low-risk pick."
        )

        left, right = st.columns([3, 2])
        with left:
            if geojson is not None:
                map_df = state_totals.copy()
                map_df["score"] = map_df["state"].map(display_scores).fillna(0)
                fig = px.choropleth(map_df, geojson=geojson, locations="state", featureidkey="properties.st_nm",
                                     color="score", color_continuous_scale="Viridis",
                                     title=f"Attractiveness Score by State \u2014 {target_sector}")
                fig.update_geos(fitbounds="locations", visible=False)
                fig.update_layout(height=420, margin=dict(l=0, r=0, t=40, b=0))
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("Map asset (india_states_map.b64) not found alongside app.py -- showing ranking only.")
        with right:
            bar_df = display_scores.head(top_n).reset_index()
            bar_df.columns = ["state", "score"]
            fig = px.bar(bar_df, x="score", y="state", orientation="h", color="score",
                         color_continuous_scale="Viridis", labels={"score": "Score (0-100)", "state": ""},
                         title=f"Top {top_n} Candidates")
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, showlegend=False, height=420)
            st.plotly_chart(fig, width='stretch')

        st.subheader(f"Why these states score the way they do")
        breakdown = norm.loc[top_states, ["scale_n", "growth_n", "fit_n"]].reset_index().melt(
            id_vars="state", var_name="component", value_name="percentile")
        breakdown["component"] = breakdown["component"].map(
            {"scale_n": "Scale", "growth_n": "Growth (size-adjusted)", "fit_n": "Sector Fit"})
        fig = px.bar(breakdown, x="state", y="percentile", color="component", barmode="group",
                     labels={"percentile": "Percentile rank among eligible states", "state": ""})
        fig.update_layout(yaxis_range=[0, 1], height=350)
        st.plotly_chart(fig, width='stretch')

        with st.expander("Full ranked table"):
            full = display_scores.head(15).rename("score").to_frame()
            full["total_startups"] = state_totals.set_index("state")["total_startups"]
            full[f"{target_sector} LQ"] = lq[target_sector]
            full["growth_%/yr"] = growth_df["trend_growth_pct"]
            full["cluster"] = features["cluster_label"]
            full["stability_%"] = top5_pct
            st.dataframe(full.round(1), width='stretch')

        summary_text = (
            f"WHERE SHOULD WE EXPAND? -- {target_sector}, excl. {home_state}\n"
            f"Generated from DPIIT data, {STABLE_YEARS[0]}-{STABLE_YEARS[-1]}\n\n"
            + "\n".join(f"{i+1}. {s}  (score {display_scores.loc[s]:.0f}/100, "
                        f"LQ {lq.loc[s, target_sector]:.2f}x, growth {growth_df.loc[s,'trend_growth_pct']:+.0f}%/yr, "
                        f"stability {top5_pct.get(s,0):.0f}%)"
                        for i, s in enumerate(top_states))
        )
        st.download_button("\U0001F4E5 Download this recommendation as text", summary_text,
                            file_name="expansion_recommendation.txt")

# ---------- TAB 2: EXPLORE A STATE ----------
with tab2:
    sel_state = st.selectbox("Pick a state", all_states, index=all_states.index(home_state))
    c1, c2 = st.columns(2)
    with c1:
        sub = (stable[stable["state"] == sel_state].groupby("industry")["startups_recognized"].sum()
               .sort_values(ascending=False).head(10).reset_index())
        fig = px.bar(sub, x="startups_recognized", y="industry", orientation="h",
                     title=f"{sel_state}: Top Industries")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
        st.plotly_chart(fig, width='stretch')
    with c2:
        trend = df[df["state"] == sel_state].groupby("year")["startups_recognized"].sum().reset_index()
        fig = px.line(trend, x="year", y="startups_recognized", markers=True, title=f"{sel_state}: Year-wise Trend")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')

    if sel_state in ranked.index:
        rank = int(ranked.rank(ascending=False)[sel_state])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Composite score", f"{ranked.loc[sel_state]:.0f}/100")
        m2.metric("Rank", f"{rank} of {len(ranked)}")
        m3.metric(f"{target_sector} LQ", f"{lq.loc[sel_state, target_sector]:.2f}x")
        m4.metric("Cluster", features.loc[sel_state, "cluster_label"])
    else:
        st.caption(f"Not scored: fewer than {min_threshold} total recognized startups -- ratios for this "
                   f"state aren't reliable enough to rank (adjust the threshold in the sidebar to include it).")

# ---------- TAB 3: COMPARE STATES ----------
with tab3:
    default_compare = top_states[:2] if len(top_states) >= 2 else all_states[:2]
    compare_states = st.multiselect("Pick 2-4 states to compare", all_states, default=default_compare,
                                     max_selections=4)
    if len(compare_states) < 2:
        st.info("Pick at least 2 states.")
    else:
        rows = []
        for s in compare_states:
            rows.append({
                "State": s,
                "Total startups": int(state_totals.set_index("state").loc[s, "total_startups"]),
                "Growth %/yr": round(growth_df.loc[s, "trend_growth_pct"], 1) if s in growth_df.index else None,
                f"{target_sector} LQ": round(lq.loc[s, target_sector], 2) if s in lq.index else None,
                "Composite score": round(ranked.loc[s], 1) if s in ranked.index else None,
                "Cluster": features.loc[s, "cluster_label"] if s in features.index else "n/a",
                "Stability %": round(top5_pct.get(s, 0.0), 0),
            })
        st.dataframe(pd.DataFrame(rows).set_index("State"), width='stretch')

        radar_states = [s for s in compare_states if s in norm.index]
        if radar_states:
            fig = go.Figure()
            for s in radar_states:
                vals = norm.loc[s, ["scale_n", "growth_n", "fit_n"]].tolist()
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]],
                                               theta=["Scale", "Growth", "Sector Fit", "Scale"],
                                               fill="toself", name=s))
            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])), height=450,
                               title="Percentile Profile Comparison")
            st.plotly_chart(fig, width='stretch')
        else:
            st.caption("None of the selected states pass the reliability threshold, so no percentile profile is available.")

# ---------- TAB 4: OPPORTUNITY VS SATURATION ----------
with tab4:
    st.markdown(
        f"A high Location Quotient can mean a strong, supportive ecosystem for **{target_sector}** -- or it can "
        f"mean the market is already crowded. Reading LQ together with how fast {target_sector} *itself* is "
        f"growing in each state tells the two apart.")
    quad_df = pd.DataFrame({
        "location_quotient": lq.loc[sector_growth.index, target_sector],
        "sector_growth_pct": sector_growth,
    })
    if len(quad_df) > 0:
        median_g = quad_df["sector_growth_pct"].median()

        def read(row):
            hi_lq, hi_g = row["location_quotient"] >= 1, row["sector_growth_pct"] >= median_g
            if hi_lq and hi_g: return "Hot but competitive"
            if hi_lq and not hi_g: return "Mature / saturated"
            if not hi_lq and hi_g: return "Emerging opportunity"
            return "Not yet a fit"

        quad_df["read"] = quad_df.apply(read, axis=1)
        fig = px.scatter(quad_df.reset_index(), x="location_quotient", y="sector_growth_pct", color="read",
                          hover_name="state", text="state",
                          title=f"{target_sector}: Specialization vs. Sector-Specific Growth",
                          labels={"location_quotient": "Location Quotient", "sector_growth_pct": "Sector growth (%/yr)"})
        fig.add_vline(x=1, line_dash="dash", line_color="grey")
        fig.add_hline(y=median_g, line_dash="dash", line_color="grey")
        fig.update_traces(textposition="top center")
        fig.update_layout(height=550)
        st.plotly_chart(fig, width='stretch')

        c1, c2 = st.columns(2)
        with c1:
            st.success("**Emerging opportunity:** " + (", ".join(quad_df[quad_df["read"] == "Emerging opportunity"].index) or "none"))
        with c2:
            st.warning("**Mature / saturated:** " + (", ".join(quad_df[quad_df["read"] == "Mature / saturated"].index) or "none"))

# ---------- TAB 5: ROBUSTNESS & NOTES ----------
with tab5:
    st.subheader("Does your selected weighting agree with a more conservative baseline?")
    robust_top = scores.drop(index=home_state, errors="ignore").nlargest(top_n, "avg_score").index.tolist()
    st.write(f"**Your selection ({weight_mode}):** {top_states}")
    st.write(f"**4-scenario robustness average** (Balanced/Growth/Scale/Fit-tilted, unweighted by your choice): {robust_top}")
    overlap = len(set(top_states) & set(robust_top))
    st.caption(f"{overlap} of {top_n} states agree between the two -- a low overlap means your specific "
               f"weighting choice is doing a lot of work in the answer, and it's worth sanity-checking why.")

    st.subheader("How much does the threshold choice matter?")
    st.caption("Re-runs the eligibility gate at each threshold (cached, so this is cheap) -- a state can newly "
               "qualify here, not just drop out, unlike a simple filter on the current set.")
    thr_results = {}
    for thr in sorted({100, 200, 500, min_threshold}):
        thr_stable, thr_totals, thr_eligible = compute_eligibility(df, STABLE_YEARS, thr)
        thr_growth = compute_growth(thr_stable, thr_totals, thr_eligible, GROWTH_YEARS)
        thr_sector_growth = compute_sector_growth(thr_stable, thr_eligible, GROWTH_YEARS, target_sector)
        thr_features = compute_clusters(thr_totals, thr_growth, state_share, thr_eligible)
        _, thr_norm, _ = compute_scores(thr_features, thr_growth, lq, thr_sector_growth, thr_totals, thr_stable,
                                         recent_years, target_sector)
        thr_score = score_with_weights(thr_norm, active_weights)
        thr_results[thr] = thr_score.drop(index=home_state, errors="ignore").nlargest(top_n).index.tolist()
    for thr, lst in thr_results.items():
        flag = "  \u2190 current" if thr == min_threshold else ""
        st.write(f"**{thr}:** {lst}{flag}")

    st.subheader("Weight-sensitivity (500 random weight draws)")
    fig = px.bar(top5_pct.head(10).reset_index(), x="pct_in_top", y="state",
                 orientation="h", labels={"pct_in_top": f"% of draws in top {top_n}"},
                 title="How often each state makes the cut, across random weightings")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=400)
    st.plotly_chart(fig, width='stretch')

    st.subheader("Methodology notes")
    st.markdown(f"""
- **Growth** is residualized against scale (a state only scores well if it's growing faster than other
  states of a *similar* size), and measured over {GROWTH_YEARS[0]}\u2013{GROWTH_YEARS[-1]} -- {STABLE_YEARS[0]} was
  excluded as DPIIT's national launch/ramp-up year.
- **States below the sidebar threshold** are excluded from scoring (not from the map/explorer) -- their
  ratios are too noisy on a small sample to rank.
- **Sector fit** blends Location Quotient with the target sector's own growth rate, so a saturated market
  doesn't automatically outscore a smaller but still-growing one.
- **The displayed score uses your sidebar weighting**, combined via a weighted geometric mean -- a state
  can't fully compensate for being weak on one dimension with strength on another. The 4-scenario average
  above is a separate, fixed robustness reference, not what's shown elsewhere in the dashboard.
- **DPIIT recognition is a registration count, not a measure of funding or revenue** -- treat this as a
  starting point for due diligence, not a substitute for it.
- Full methodology, sensitivity checks and derivation: see the companion analysis notebook.
""")
