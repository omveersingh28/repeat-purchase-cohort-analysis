"""Repeat Purchase Analysis by Cohort — Streamlit dashboard.

Reads ONLY the small aggregates in `app_data/` (written by `python -m src.run_pipeline`), never
the raw dataset or `data/processed/`. That is what makes it deployable without the 97 MB file.
Every number shown comes from `app_data/results.json` or the CSVs next to it.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
APP_DATA = ROOT / "app_data"
sys.path.insert(0, str(ROOT / "src"))

from retention import core, theme  # noqa: E402  (pure pandas/numpy helpers and shared colours)

st.set_page_config(page_title="Repeat Purchase Analysis by Cohort", page_icon="📦", layout="wide")

PLOT_CONFIG = {"displayModeBar": False, "responsive": True}
MAX_SERIES = len(theme.SERIES)


# --------------------------------------------------------------------------- data loading
@st.cache_data
def load_results() -> dict:
    return json.loads((APP_DATA / "results.json").read_text())


@st.cache_data
def load_csv(name: str, index_col: str | None = None) -> pd.DataFrame:
    return pd.read_csv(APP_DATA / name, index_col=index_col)


@st.cache_data
def load_matrix(name: str) -> pd.DataFrame:
    """Cohort x month-index matrix with string cohort labels; all-empty columns dropped."""
    m = pd.read_csv(APP_DATA / name, index_col="CohortMonth")
    m.index = m.index.astype(str)
    return m.dropna(axis=1, how="all")


# --------------------------------------------------------------------------- formatting
def pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or x != x else f"{x * 100:.{digits}f}%"


def gbp(x: float, digits: int = 0) -> str:
    return f"£{x:,.{digits}f}"


def month_name(ym: str) -> str:
    """'2009-12' -> 'Dec 2009'."""
    return pd.Period(ym, freq="M").strftime("%b %Y")


def style(fig: go.Figure, height: int = 420, legend: bool = True) -> go.Figure:
    """One explicit look for every chart (matches the Matplotlib figures in the README)."""
    fig.update_layout(
        height=height, font={"family": theme.FONT_CSS, "size": 13, "color": theme.INK},
        paper_bgcolor=theme.SURFACE, plot_bgcolor=theme.SURFACE,
        margin={"l": 8, "r": 8, "t": 36 if legend else 12, "b": 8}, showlegend=legend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0, "font": {"color": theme.INK_SECONDARY}},
        hoverlabel={"font": {"family": theme.FONT_CSS}}, bargap=0.28)
    axis = {"gridcolor": theme.GRID, "linecolor": theme.AXIS, "zeroline": False, "automargin": True,
            "tickfont": {"color": theme.INK_SECONDARY}, "title_font": {"color": theme.INK_SECONDARY}}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def show(fig: go.Figure, key: str) -> None:
    st.plotly_chart(fig, width="stretch", theme=None, config=PLOT_CONFIG, key=key)


def table(df: pd.DataFrame, label: str = "Show the data as a table", **kw) -> None:
    with st.expander(label):
        st.dataframe(df, width="stretch", **kw)


# --------------------------------------------------------------------------- load everything
R = load_results()
clean, bounds, ret, rep, rev = R["cleaning"], R["bounds"], R["retention"], R["repeat"], R["revenue"]
segs, gaps, sens = R["segments"], R["segment_gaps"], R["sensitivity"]
W = rep["window_days"]
WCOL, SEGCOL = f"Repeat{W}d", f"repeat_{W}d"
FIRST = bounds["first_month"]
FIRST_NAME = month_name(FIRST)
km_r = {int(k): v for k, v in rep["km_repeated_by"].items() if v is not None}

summary = load_csv("cohort_summary.csv", "CohortMonth")
summary.index = summary.index.astype(str)
sizes = load_csv("cohort_sizes.csv", "CohortMonth")
sizes.index = sizes.index.astype(str)
avg_curve = load_csv("avg_retention_curve.csv")
km = load_csv("survival_to_second.csv")
hist = load_csv("days_to_second_hist.csv")
split = load_csv("revenue_split.csv", "CohortMonth")
split.index = split.index.astype(str)
segments = load_csv("segments.csv")
segments["segment_value"] = segments["segment_value"].astype(str)

# --------------------------------------------------------------------------- header
st.title("Repeat Purchase Analysis by Cohort")
st.markdown(
    "The business acquires new customers every month, but does not know how many of them come back to buy "
    "again, or how quickly. This dashboard groups customers into **cohorts by first-purchase month** and "
    "measures **repeat-purchase behaviour over time** - which cohorts retain best, when customers stop "
    "coming back, and where retention effort should focus. Data: *Online Retail II*, a UK online gift "
    f"retailer, {pd.Timestamp(clean['date_min']):%d %b %Y} to {pd.Timestamp(clean['date_max']):%d %b %Y}.")

kpis = [
    ("Customers", f"{clean['customers']:,}", "Identified customers after cleaning (guest checkouts cannot be tracked)."),
    ("Orders", f"{clean['invoices']:,}", "Distinct invoices after cleaning."),
    (f"Repeat within {W} days", pct(rep["repeat_window_mean"]),
     f"Mean across the {ret['n_fully_observed']} cohorts whose full {W}-day window is observed. "
     f"True-new cohorts only: {pct(rep['repeat_window_mean_true_new'])}."),
    ("Median days to 2nd order", f"{rep['median_days_to_second']:.0f}", "Among customers with at least two orders."),
    ("Month-1 retention", pct(ret["avg_curve"]["1"]),
     f"Average share of a true-new cohort ordering again the month after the first order ({FIRST_NAME} excluded)."),
    ("Repeat share of revenue", pct(rev["repeat_share"]), "Revenue from 2nd and later orders / all revenue (gross of returns)."),
]
for row in (kpis[:3], kpis[3:]):
    for col, (label, value, help_text) in zip(st.columns(3), row):
        col.metric(label, value, help=help_text, border=True)

tabs = st.tabs(["Retention", "Cohort comparison", "Time to second purchase", "Revenue", "Segments",
                "Insights & recommendations", "Method & caveats"])

# =========================================================================== 1. Retention
with tabs[0]:
    st.subheader(f"Only about {pct(ret['avg_curve']['1'], 0)} of a new cohort orders again the following month "
                 "- and the rate barely decays after that")
    c1, c2 = st.columns([2, 1])
    segment = c1.radio("Customers", ["All", "UK", "International"], horizontal=True, key="ret_segment")
    show_values = c2.toggle("Show values in cells", value=True, key="ret_values",
                            help="Turn off on a narrow screen - hover or tap a cell for its value instead.")
    suffix = {"All": "", "UK": "_uk", "International": "_intl"}[segment]
    P, C = load_matrix(f"retention_pct{suffix}.csv"), load_matrix(f"retention_counts{suffix}.csv")
    C = C.reindex(index=P.index, columns=P.columns)
    cohort_n = C.iloc[:, 0]

    z = P.to_numpy(dtype=float)
    body = z.copy()
    body[:, 0] = np.nan                               # month 0 is 100% by definition: keep it off the colour scale
    month0 = np.full_like(z, np.nan)
    month0[:, 0] = z[:, 0]
    counts = np.dstack([C.to_numpy(dtype=float), np.repeat(cohort_n.to_numpy(dtype=float)[:, None], z.shape[1], 1)])
    labels = [f"{c} †" if c == FIRST else c for c in P.index]
    text = [["" if np.isnan(v) else f"{v:.0f}" for v in row] for row in z]
    hover = ("Cohort %{y}<br>Month %{x}<br><b>%{z:.1f}% active</b><br>"
             "%{customdata[0]:.0f} of %{customdata[1]:.0f} customers<extra></extra>")
    scale = [[i / (len(theme.SEQUENTIAL) - 1), c] for i, c in enumerate(theme.SEQUENTIAL)]
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=body, x=list(P.columns), y=labels, customdata=counts, colorscale=scale, zmin=0,
        zmax=float(np.nanmax(body)), xgap=2, ygap=2, hoverongaps=False, hovertemplate=hover,
        colorbar={"title": {"text": "% of cohort active", "side": "top"}, "ticksuffix": "%", "thickness": 10,
                  "orientation": "h", "y": -0.13, "yanchor": "top", "len": 0.6, "x": 0.5, "xanchor": "center"}))
    fig.add_trace(go.Heatmap(
        z=month0, x=list(P.columns), y=labels, customdata=counts, colorscale=[[0, "#eeede8"], [1, "#eeede8"]],
        showscale=False, xgap=2, ygap=2, hoverongaps=False, hovertemplate=hover))
    if show_values:                                   # explicit ink per cell so every value stays legible
        dark = float(np.nanmax(body)) * 0.5
        fig.update_layout(annotations=[
            {"x": col, "y": labels[i], "text": text[i][j], "showarrow": False,
             "font": {"size": 10, "color": theme.MUTED if j == 0 else ("#ffffff" if z[i, j] > dark else theme.INK)}}
            for i in range(z.shape[0]) for j, col in enumerate(P.columns) if text[i][j]])
    style(fig, height=max(460, 26 * len(P) + 150), legend=False)
    fig.update_xaxes(type="category", title_text="Months since first order (cohort index)", showgrid=False)
    fig.update_yaxes(type="category", autorange="reversed", title_text="Cohort (month of first order)",
                     showgrid=False, tickmode="array", tickvals=labels)
    show(fig, "heatmap")
    st.caption(
        f"Each cell is the share of a cohort that placed at least one order in that month; hover for customer "
        f"counts. **Blank cells are months not yet fully observed** (the data ends "
        f"{pd.Timestamp(bounds['data_end']):%d %b %Y}, so {month_name(bounds['last_complete_month'])} is the last "
        f"complete month) - they are missing, not zero. **† {FIRST_NAME} is not a real cohort**: the data starts "
        f"that month, so its 'new' customers include long-standing ones; it is shown for context and excluded "
        f"from every average. Month 0 is 100% by definition and drawn in grey.")
    if segment == "International":
        st.warning(f"International cohorts are small (median {cohort_n.median():.0f} customers per cohort), so "
                   "individual cells swing widely. Compare segments on the pooled metrics in the Segments tab.")
    table(P.round(1), "Show retention (%) as a table")

    st.subheader(f"{pct(1 - ret['avg_curve']['1'], 0)} of new customers are not active the month after their "
                 "first order")
    drop = ret["biggest_dropoff"]
    fig = go.Figure(go.Scatter(
        x=avg_curve["month_index"], y=avg_curve["mean_retention"] * 100, mode="lines+markers",
        line={"color": theme.PRIMARY, "width": 2}, marker={"size": 8, "line": {"color": theme.SURFACE, "width": 2}},
        customdata=avg_curve["n_cohorts"], name="Average retention",
        hovertemplate="Month %{x}<br><b>%{y:.1f}% active</b><br>averaged over %{customdata} cohorts<extra></extra>"))
    style(fig, height=360, legend=False)
    fig.update_xaxes(title_text="Months since first order (cohort index)", dtick=3)
    fig.update_yaxes(title_text="Mean share of cohort active (%)", ticksuffix="%", rangemode="tozero")
    show(fig, "avg_curve")
    st.caption(
        f"Average over true-new cohorts, using only cohorts that have fully observed each month (at least three). "
        f"The biggest drop-off is month {drop['month_index'] - 1} → {drop['month_index']} "
        f"(**{drop['change_pp']:+.1f} pp**). Month 2 is {pct(ret['avg_curve']['2'])}, slightly above month 1 "
        f"({pct(ret['avg_curve']['1'])}) - a small rebound that is shown as it is, not smoothed.")
    table(avg_curve.assign(mean_retention=(avg_curve["mean_retention"] * 100).round(2)), hide_index=True)

# =========================================================================== 2. Cohort comparison
with tabs[1]:
    P = load_matrix("retention_pct.csv")
    per_index = pd.PeriodIndex(summary.index, freq="M")
    default = [str(c) for c in core.select_cohorts(
        summary.set_axis(per_index), {"first_month": pd.Period(FIRST, freq="M")}, W)]
    st.subheader("Retention curves by cohort")
    chosen = st.multiselect(f"Cohorts to compare (up to {MAX_SERIES})", list(P.index), default=default,
                            max_selections=MAX_SERIES, format_func=lambda c: f"{c} †" if c == FIRST else c,
                            key="cohort_pick")
    # colour follows the cohort, not its position: a cohort keeps its colour while it stays selected
    slots: dict = st.session_state.setdefault("cohort_slots", {})
    for gone in [c for c in slots if c not in chosen]:
        del slots[gone]
    for c in chosen:
        if c not in slots:
            slots[c] = next(i for i in range(MAX_SERIES) if i not in slots.values())
    if chosen:
        fig = go.Figure()
        for c in sorted(chosen):
            s = P.loc[c].drop(labels=P.columns[0]).dropna()
            fig.add_trace(go.Scatter(
                x=[int(i) for i in s.index], y=s.values, mode="lines+markers", name=f"{month_name(c)}"
                + (" †" if c == FIRST else ""), line={"color": theme.SERIES[slots[c]], "width": 2},
                marker={"size": 7, "line": {"color": theme.SURFACE, "width": 1.5}},
                hovertemplate=f"{month_name(c)} cohort (n={int(sizes.loc[c, 'CohortSize']):,})<br>Month %{{x}}"
                              "<br><b>%{y:.1f}% active</b><extra></extra>"))
        style(fig, height=400)
        fig.update_xaxes(title_text="Months since first order (cohort index)", dtick=3)
        fig.update_yaxes(title_text="Share of cohort active (%)", ticksuffix="%", rangemode="tozero")
        show(fig, "cohort_curves")
        st.caption("Month 0 (100% by definition) is omitted so differences are visible. Each line stops at the last "
                   f"fully observed month. † {FIRST_NAME} includes pre-existing customers. Single cohorts are "
                   "noisy - read the level, not each wiggle.")
        table(P.loc[sorted(chosen)].round(1))
    else:
        st.info("Select at least one cohort.")

    full = summary[summary["FullyObserved"]]
    true_new = full[full.index != FIRST]
    st.subheader(f"Like-for-like, {pct(rep['repeat_window_min_true_new'], 0)} to "
                 f"{pct(rep['repeat_window_max_true_new'], 0)} of a new cohort re-orders within {W} days")
    fig = go.Figure(go.Bar(
        x=list(full.index), y=full[WCOL] * 100, customdata=full["CohortSize"],
        marker_color=[theme.DEEMPHASIS if c == FIRST else theme.PRIMARY for c in full.index],
        hovertemplate="Cohort %{x} (n=%{customdata:,})<br><b>%{y:.1f}%</b> re-ordered within "
                      f"{W} days<extra></extra>"))
    fig.update_traces(name=f"Repeat within {W} days", showlegend=False)
    fig.add_trace(go.Scatter(
        x=list(full.index), y=[rep["repeat_window_mean"] * 100] * len(full), mode="lines", hoverinfo="skip",
        line={"color": theme.INK, "width": 1.2, "dash": "dash"},
        name=f"Mean of fully observed cohorts: {pct(rep['repeat_window_mean'])}"))
    style(fig, height=380)
    fig.update_xaxes(type="category", title_text="Cohort (month of first order)", tickangle=-45, showgrid=False)
    fig.update_yaxes(title_text=f"2nd order within {W} days (%)", ticksuffix="%", rangemode="tozero")
    show(fig, "repeat_bar")
    st.caption(
        f"Only the {ret['n_fully_observed']} of {ret['n_cohorts']} cohorts whose full {W}-day window lies inside "
        f"the data are shown, so every customer had the same opportunity. The grey bar is {FIRST_NAME} "
        f"(pre-existing customers). Mean of all fully observed cohorts: **{pct(rep['repeat_window_mean'])}**; "
        f"true-new cohorts only: **{pct(rep['repeat_window_mean_true_new'])}**. Weakest true-new cohort: "
        f"{true_new[WCOL].idxmin()} ({pct(true_new[WCOL].min())}, {int(true_new.loc[true_new[WCOL].idxmin(), 'CohortSize'])} "
        "customers).")
    st.info(
        f"**Why not just compare '% who ever repeated'?** That number falls from "
        f"{pct(rep['repeat_ever_first_cohort'])} for the first cohort to {pct(rep['repeat_ever_last_cohort'])} for "
        "the last simply because newer cohorts have been watched for less time. It measures observation time, "
        "not customer quality, so cohorts are never ranked on it here.")
    table(summary[["CohortSize", "RepeatEver", WCOL, f"MedianDaysToSecond_{W}d", "FullyObserved"]]
          .assign(RepeatEver=lambda d: (d["RepeatEver"] * 100).round(1), **{WCOL: lambda d: (d[WCOL] * 100).round(1)}))

# =========================================================================== 3. Time to second purchase
with tabs[2]:
    lead = W if W in km_r else sorted(km_r)[len(km_r) // 2]
    st.subheader(f"{pct(km_r[lead], 0)} of customers place a second order within {lead} days, "
                 f"{pct(km_r[max(km_r)], 0)} within {max(km_r)} days")
    fig = go.Figure(go.Scatter(
        x=km["day"], y=km["repeated_by"] * 100, mode="lines", line={"color": theme.PRIMARY, "width": 2, "shape": "hv"},
        customdata=km["at_risk"], name="Kaplan-Meier estimate",
        hovertemplate="Day %{x}<br><b>%{y:.1f}%</b> have re-ordered<br>%{customdata:,} customers still at risk"
                      "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=list(km_r), y=[v * 100 for v in km_r.values()], mode="markers+text",
        text=[f"day {d}: {pct(v)}" for d, v in km_r.items()], textposition="bottom right",
        textfont={"color": theme.INK, "size": 12}, hoverinfo="skip", cliponaxis=False,
        marker={"size": 10, "color": theme.PRIMARY, "line": {"color": theme.SURFACE, "width": 2}}))
    style(fig, height=420, legend=False)
    fig.update_xaxes(title_text="Days since first order", range=[-5, km["day"].max() * 1.16])
    fig.update_yaxes(title_text="Customers with a 2nd order (%)", ticksuffix="%", range=[0, 100])
    show(fig, "km")
    st.caption(
        f"Kaplan-Meier estimate over all {rep['n_customers']:,} identified customers. Customers who have not yet "
        "re-ordered are *censored* at the end of the data instead of being counted as lost, so recent cohorts do "
        f"not bias the curve. Note this reads {pct(km_r.get(W))} at day {W} while the cohort metric averages "
        f"{pct(rep['repeat_window_mean'])}: the curve pools every customer (including the large {FIRST_NAME} "
        "cohort) and weights by customer; the cohort metric weights cohorts equally and leaves out incomplete "
        "windows. Both are correct - see Method & caveats.")
    table(km.assign(repeated_by=(km["repeated_by"] * 100).round(2)), hide_index=True)

    st.subheader(f"Half of repeat customers come back within {rep['median_days_to_second']:.0f} days")
    day0, body, tail = hist.iloc[0], hist.iloc[1:-1], hist.iloc[-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=body["label"], y=body["customers"], marker_color=theme.PRIMARY, name="7-day bins",
                         hovertemplate="Days %{x}<br><b>%{y:,}</b> customers<extra></extra>"))
    fig.add_trace(go.Bar(x=[day0["label"]], y=[day0["customers"]], marker_color=theme.ACCENT,
                         name="Day 0 (same-day second invoice)",
                         hovertemplate="Same day<br><b>%{y:,}</b> customers<extra></extra>"))
    fig.add_trace(go.Bar(x=[tail["label"]], y=[tail["customers"]], marker_color=theme.DEEMPHASIS,
                         name=f"{tail['label']} days (pooled)",
                         hovertemplate=f"{tail['label']} days<br><b>%{{y:,}}</b> customers<extra></extra>"))
    style(fig, height=380)
    fig.update_layout(bargap=0.12)
    fig.update_xaxes(type="category", categoryorder="array", categoryarray=list(hist["label"]), nticks=14,
                     title_text="Days between first and second order", showgrid=False, tickangle=-45)
    fig.update_yaxes(title_text="Repeat customers")
    show(fig, "hist")
    st.caption(
        f"{rep['n_repeat_customers']:,} customers with at least two orders. **{pct(rep['same_day_share'])} placed "
        "their second invoice on the same day as the first** - often split or corrected orders rather than genuine "
        "re-purchases. They are kept, but bear it in mind when reading the median. The day-0 bar is a single "
        "day; the others are 7-day bins.")
    table(hist, hide_index=True)

# =========================================================================== 4. Revenue
with tabs[3]:
    st.subheader(f"Repeat orders bring in {pct(rev['repeat_share'], 0)} of all revenue")
    include_first = st.checkbox(
        f"Include the {FIRST_NAME} cohort (pre-existing customers, {gbp(split.loc[FIRST, 'Total'] / 1e6, 1)}M - "
        "it dwarfs every other bar)", value=False, key="rev_first")
    s = split if include_first else split[split.index != FIRST]
    fig = go.Figure()
    for column, colour, name in (("First", theme.PRIMARY, "First orders"),
                                 ("Repeat", theme.ACCENT, "Repeat orders (2nd onwards)")):
        fig.add_trace(go.Bar(x=list(s.index), y=s[column] / 1e3, name=name, marker_color=colour,
                             marker_line={"color": theme.SURFACE, "width": 1.5},
                             hovertemplate="Cohort %{x}<br>" + name + ": <b>£%{y:,.0f}k</b><extra></extra>"))
    style(fig, height=400)
    fig.update_layout(barmode="stack")
    fig.update_xaxes(type="category", title_text="Cohort (month of first order)", tickangle=-45, showgrid=False)
    fig.update_yaxes(title_text="Revenue to date (£ thousand)", tickprefix="£", ticksuffix="k")
    show(fig, "rev_bars")

    st.subheader("Repeat share of each cohort's revenue")
    fig = go.Figure(go.Scatter(
        x=list(split.index), y=split["RepeatShare"] * 100, mode="lines+markers",
        line={"color": theme.ACCENT, "width": 2}, marker={"size": 8, "line": {"color": theme.SURFACE, "width": 2}},
        hovertemplate="Cohort %{x}<br><b>%{y:.1f}%</b> of revenue from repeat orders<extra></extra>"))
    style(fig, height=320, legend=False)
    fig.update_xaxes(type="category", title_text="Cohort (month of first order)", tickangle=-45)
    fig.update_yaxes(title_text="Repeat share (%)", ticksuffix="%", range=[0, 100])
    show(fig, "rev_share")
    st.caption(
        f"First orders: {gbp(rev['first_total'])}; repeat orders: {gbp(rev['repeat_total'])}. Revenue is **gross "
        "of returns** (cancellations are dropped, not netted off). Newer cohorts have had less time to re-order, so "
        "their lower repeat share is an observation-time effect, not a quality signal. Repeat revenue is "
        f"concentrated: the median repeat customer generated {gbp(rev['median_repeat_revenue_per_repeat_customer'])} "
        f"from repeat orders, the mean {gbp(rev['mean_repeat_revenue_per_repeat_customer'])} (wholesale buyers).")
    table(split.round({"First": 0, "Repeat": 0, "Total": 0, "RepeatShare": 3}))

# =========================================================================== 5. Segments
with tabs[4]:
    spread = {k: (g[SEGCOL].max() - g[SEGCOL].min()) * 100 for k, g in segments.groupby("segment_type", sort=False)}
    widest, narrowest = max(spread, key=spread.get), min(spread, key=spread.get)
    st.subheader(f"{theme.SEGMENT_LABELS[widest]} separates repeaters most ({spread[widest]:.1f} pp spread); "
                 f"{theme.SEGMENT_LABELS[narrowest].lower()} barely matters ({spread[narrowest]:.1f} pp)")
    st.caption(f"Repeat-within-{W}-days, true-new customers in fully observed cohorts only, so every customer had "
               f"the same {W} days to return. Sample sizes are shown on every bar; groups under 100 customers "
               f"are flagged. The Q4 comparison rests on {len(rep['q4_cohorts_fully_observed'])} fully observed Q4 "
               f"cohorts ({', '.join(rep['q4_cohorts_fully_observed'])}) - a single season.")
    for seg_type, g in segments.groupby("segment_type", sort=False):
        g = g.assign(label=[theme.segment_value_label(seg_type, v) for v in g["segment_value"]])
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**{theme.SEGMENT_LABELS[seg_type]}**")
            fig = go.Figure(go.Bar(
                y=g["label"], x=g[SEGCOL] * 100, orientation="h", marker_color=theme.PRIMARY, customdata=g["n"],
                text=[f"{pct(v)}  ·  n={n:,}" + ("  ⚠ small sample" if small else "")
                      for v, n, small in zip(g[SEGCOL], g["n"], g["small_sample"])],
                textposition="outside", textfont={"color": theme.INK}, cliponaxis=False,
                hovertemplate="%{y}<br><b>%{x:.1f}%</b> re-ordered within " + f"{W} days<br>n = %{{customdata:,}}"
                              "<extra></extra>"))
            style(fig, height=70 + 46 * len(g), legend=False)
            fig.update_xaxes(range=[0, segments[SEGCOL].max() * 100 * 1.55], ticksuffix="%",
                             title_text=f"2nd order within {W} days (%)")
            fig.update_yaxes(type="category", autorange="reversed", showgrid=False)
            show(fig, f"seg_{seg_type}")
        with right:
            gap = gaps.get(seg_type)
            if gap:
                a = theme.segment_value_label(seg_type, gap["a"])
                b = theme.segment_value_label(seg_type, gap["b"])
                verdict = ("a real difference" if gap["distinguishable"]
                           else "**no meaningful difference** - the interval spans zero")
                st.markdown(f"{a} vs {b}: **{gap['diff_pp']:+.1f} pp** (95% CI {gap['ci_low_pp']:+.1f} to "
                            f"{gap['ci_high_pp']:+.1f}) → {verdict}.")
            st.dataframe(
                g[["label", "n", SEGCOL, "avg_orders", "median_first_order_value", "small_sample"]]
                .rename(columns={"label": "Segment", SEGCOL: f"Repeat {W}d (%)", "avg_orders": "Avg orders",
                                 "median_first_order_value": "Median first order (£)", "small_sample": "Small sample"})
                .assign(**{f"Repeat {W}d (%)": lambda d: (d[f"Repeat {W}d (%)"] * 100).round(1),
                           "Avg orders": lambda d: d["Avg orders"].round(2),
                           "Median first order (£)": lambda d: d["Median first order (£)"].round(0)}),
                hide_index=True, width="stretch")
    if segments["small_sample"].any():
        st.warning("Segments marked ⚠ have fewer than 100 customers - treat their rates as indicative only.")
    else:
        st.caption("No segment here has fewer than 100 customers.")

# =========================================================================== 6. Insights & recommendations
with tabs[5]:
    tiers, q4 = segs["FirstOrderTier"], segs["IsQ4Cohort"]
    drop = ret["biggest_dropoff"]
    st.subheader("Key findings")
    st.markdown(f"""
1. **The lifecycle problem is the first month.** Average retention falls **{abs(drop['change_pp']):.1f} pp** between
   month {drop['month_index'] - 1} and month {drop['month_index']}, to **{pct(ret['avg_curve']['1'])}**, then holds
   roughly level (month 2: {pct(ret['avg_curve']['2'])}).
2. **{pct(rep['repeat_window_mean'], 0)} of new customers re-order within {W} days.** Like-for-like
   repeat-within-{W}-days averages **{pct(rep['repeat_window_mean'])}** across fully observed cohorts ({pct(rep['repeat_window_mean_true_new'])} for
   true-new cohorts), with a median of **{rep['median_days_to_second']:.0f} days** to the second order.
3. **Newer cohorts are neither better nor worse.** {gaps['CohortYear']['a']} vs {gaps['CohortYear']['b']} cohorts
   differ by {gaps['CohortYear']['diff_pp']:+.1f} pp (95% CI {gaps['CohortYear']['ci_low_pp']:+.1f} to
   {gaps['CohortYear']['ci_high_pp']:+.1f}) - indistinguishable from zero.
4. **Repeat orders are the business.** They generate **{pct(rev['repeat_share'])}** of revenue.
5. **First-order value and season matter; region does not.** Repeat rate climbs from {pct(tiers['Low'][SEGCOL])}
   (low first order) to {pct(tiers['High'][SEGCOL])} (high); Q4-acquired customers repeat at
   {pct(q4['True'][SEGCOL])} vs {pct(q4['False'][SEGCOL])}; the UK vs international gap of
   {gaps['Region']['diff_pp']:+.1f} pp is not a real difference.
""")
    st.subheader("Recommendations with quantified impact")
    st.warning(
        "Read the estimates as **orders of magnitude, not forecasts**. Uplift assumptions are illustrative (not "
        "measured), values use the **median** repeat revenue because wholesale buyers inflate the mean, everything "
        "is gross revenue (no cost, margin or channel data exists), and the estimates overlap so they must not be "
        "added together.")

    def fmt_input(k: str, v) -> str:
        if any(w in k for w in ("rate", "share", "uplift")):
            return f"{v * 100:.1f}%" if "uplift" not in k else f"{v * 100:.0f} pp"
        if "revenue" in k or "value" in k:
            return gbp(v, 2)
        return f"{v:,.2f}" if isinstance(v, float) else f"{v:,}"

    for i, r in enumerate(R["recommendations"], 1):
        with st.container(border=True):
            st.markdown(f"#### {i}. {r['title']}")
            left, right = st.columns([3, 1])
            left.markdown(f"**Finding.** {r['finding']}\n\n**Action.** {r['action']}")
            right.metric("Illustrative estimate", f"≈ {gbp(r['estimate_gbp_per_year'] / 1000)}k / year")
            st.markdown("**The arithmetic**")
            st.code(r["formula"].replace(" x ", " × "), language=None)
            st.dataframe(pd.DataFrame({"input": list(r["inputs"]),
                                       "value": [fmt_input(k, v) for k, v in r["inputs"].items()]}),
                         hide_index=True, width="stretch")
            st.markdown(f"= **{gbp(r['estimate_gbp_per_year'])} per year**")
            st.caption(f"Assumptions: {r['assumptions']}")

# =========================================================================== 7. Method & caveats
with tabs[6]:
    st.subheader("Definitions")
    st.table(pd.DataFrame([
        ("Order", "One distinct `Invoice` (its line items summed)"),
        ("Cohort", "Calendar month of the customer's first order"),
        ("Cohort index", "Whole months between an order's month and the cohort month (0, 1, 2, …)"),
        ("Active in month N", "Customer placed ≥ 1 order at cohort index N"),
        ("Retention rate (month N)", "Customers of the cohort active in month N ÷ cohort size"),
        ("Repeat purchaser", "Customer with ≥ 2 orders"),
        (f"Repeat-within-{W}-days", f"2nd order placed ≤ {W} days after the 1st ÷ cohort size — the headline metric"),
        ("Repeat-ever", "≥ 2 orders at any point ÷ cohort size — biased by observation time, context only"),
        ("Days to second purchase", "Calendar days between first and second order dates"),
        ("First vs repeat revenue", "Revenue from order #1 vs orders #2+"),
        ("True-new cohort", "Any cohort after the first month in the data"),
    ], columns=["Term", "Definition"]).set_index("Term"))

    st.subheader("Cleaning waterfall")
    steps = [("Raw rows", "raw_rows"), ("Has a CustomerID", "after_missing_customer"),
             ("Not a cancellation / adjustment", "after_cancel_adjust"),
             ("Quantity > 0 and Price > 0", "after_qty_price"), ("Product stock code (final)", "after_non_product")]
    wf = pd.DataFrame({"Step": [s for s, _ in steps], "Rows remaining": [clean[k] for _, k in steps]})
    wf["Rows removed"] = (-wf["Rows remaining"].diff()).fillna(0).astype(int)
    wf["% of raw removed"] = (wf["Rows removed"] / clean["raw_rows"] * 100).round(2)
    st.dataframe(wf, hide_index=True, width="stretch")

    st.subheader("Nine data traps and how each is handled")
    traps = [
        ("Left-censoring: the first cohort is not a real cohort",
         f"The data begins in {FIRST_NAME}, so everyone who bought that month is recorded as 'new', including "
         f"long-standing customers ({pct(rep['repeat_ever_first_cohort'])} repeat-ever vs "
         f"{pct(rep['repeat_ever_last_cohort'])} for the last cohort). It is flagged, excluded from the retention "
         "averages and segment comparisons, and shown in charts only with an explicit label."),
        ("Right-censoring: recent cohorts have had less time",
         f"'% who ever repeated' mostly measures how long a cohort has been watched. Cohorts are compared on "
         f"repeat-within-{W}-days, computed only for the {ret['n_fully_observed']} of {ret['n_cohorts']} cohorts "
         "whose whole window is inside the data, and on a Kaplan-Meier curve that censors non-repeaters correctly."),
        ("Partial final month",
         f"The data ends {pd.Timestamp(bounds['data_end']):%d %b %Y}. The last complete month "
         f"({month_name(bounds['last_complete_month'])}) is derived in code and later retention cells are blank, "
         "never zero."),
        ("Non-product lines",
         f"Postage, fees, manual adjustments, samples and test rows ({', '.join(clean['non_product_codes'])}) are "
         f"excluded: {clean['non_product_rows']:,} rows, {len(clean['non_product_codes'])} codes, "
         f"{pct(clean['non_product_revenue_share'], 2)} of revenue, {clean['customers_lost_to_non_product']} "
         f"customers lost entirely. Sensitivity check: keeping them moves repeat-within-{W}-days by "
         f"{sens['repeat_window_mean_diff_pp']:+.2f} pp - "
         f"{'material' if sens['material'] else 'immaterial'} (threshold {sens['threshold_pp']:.0f} pp)."),
        ("Cancellations and adjustments are dropped, not netted off",
         "Reported revenue is therefore gross of returns."),
        ("Same-day second invoices",
         f"{pct(rep['same_day_share'])} of repeat customers place their second order on day 0 - often split or "
         "corrected orders. They are kept, and the share is reported wherever the median is discussed."),
        ("Missing customer IDs",
         f"{clean['missing_customer_rows']:,} rows ({pct(clean['missing_customer_share'])}) have no CustomerID "
         "(guest checkouts) and cannot be tracked. Retention is measured on identified customers only."),
        ("Wholesale buyers skew means",
         f"Repeat revenue per repeat customer: mean {gbp(rev['mean_repeat_revenue_per_repeat_customer'])} vs median "
         f"{gbp(rev['median_repeat_revenue_per_repeat_customer'])}. Medians are used for typical behaviour and "
         "for every impact estimate."),
        ("Two valid repeat numbers disagree",
         f"Cohort repeat-within-{W}-days averages {pct(rep['repeat_window_mean'])}; the Kaplan-Meier curve reads "
         f"{pct(km_r.get(W))} at day {W}. KM pools all customers and weights by customer, so the large "
         f"{FIRST_NAME} cohort pulls it up; the cohort metric weights cohorts equally and leaves out incomplete "
         "windows. Both are reported, with the reason."),
    ]
    for i, (title, body_text) in enumerate(traps, 1):
        with st.expander(f"{i}. {title}", expanded=i <= 2):
            st.markdown(body_text)

    st.subheader("Limitations")
    st.markdown(
        "- Revenue is gross of returns.\n- Many customers are wholesalers, not consumers; values are heavily skewed.\n"
        f"- The {FIRST_NAME} cohort is left-censored.\n- Guest checkouts are untracked.\n"
        "- No marketing, cost, margin or channel data.\n- Segment differences are correlations, not causal effects.\n"
        "- All uplift figures are illustrative.")

st.divider()
st.caption(f"Source: Chen, D. *Online Retail II* [Dataset]. UCI Machine Learning Repository. · "
           f"Results generated {R['generated_at']} by `python -m src.run_pipeline` · dashboard reads `app_data/` only.")
