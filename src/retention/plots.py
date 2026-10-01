"""src/retention/plots.py — every figure in the project.

Each function takes already-computed data (the outputs of `core`) and returns a Matplotlib
`Figure`; nothing here reads files or computes a metric that `core` does not already define.
Titles state the finding and are built from the data passed in, so no number is hard-coded.
"""
from __future__ import annotations

import textwrap

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from . import theme as t

DPI = 160

RC = {
    "font.family": "sans-serif", "font.sans-serif": t.FONT_FAMILY, "font.size": 10,
    "figure.facecolor": t.SURFACE, "axes.facecolor": t.SURFACE, "savefig.facecolor": t.SURFACE,
    "text.color": t.INK, "axes.labelcolor": t.INK_SECONDARY, "axes.edgecolor": t.AXIS,
    "xtick.color": t.INK_SECONDARY, "ytick.color": t.INK_SECONDARY,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.labelsize": 10,
    "axes.grid": True, "grid.color": t.GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2.0, "lines.markersize": 5,
}

#: Applied once at import so every figure (pipeline, notebooks) shares one explicit style.
matplotlib.rcParams.update(RC)

PCT = mticker.PercentFormatter(1.0, decimals=0)
THOUSANDS = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")
SEQ_CMAP = LinearSegmentedColormap.from_list("retention_blue", t.SEQUENTIAL)
SEQ_CMAP.set_bad(t.SURFACE)


# --------------------------------------------------------------------------- helpers
def _figure(width: float, height: float, nrows: int = 1, ncols: int = 1, **kw):
    """Create a styled Figure without touching pyplot's global state."""
    fig = Figure(figsize=(width, height), dpi=DPI, facecolor=t.SURFACE, layout="constrained")
    axes = fig.subplots(nrows, ncols, **kw)
    return fig, axes


def _titles(ax, title: str, subtitle: str | None = None, wrap: int = 88) -> None:
    """Left-aligned finding-as-title with an optional muted subtitle beneath it."""
    title = "\n".join(textwrap.wrap(title, wrap))
    pad = 10
    if subtitle:
        subtitle = "\n".join(textwrap.wrap(subtitle, int(wrap * 1.3)))
        ax.annotate(subtitle, xy=(0, 1), xycoords="axes fraction", xytext=(0, 8),
                    textcoords="offset points", ha="left", va="bottom", fontsize=9.5,
                    color=t.INK_SECONDARY, annotation_clip=False)
        pad = 14 + 13 * (subtitle.count("\n") + 1)
    ax.set_title(title, loc="left", pad=pad, fontsize=13, fontweight="bold", color=t.INK)


def _month_label(p) -> str:
    """Period('2010-01') -> 'Jan 2010'."""
    return p.strftime("%b %Y")


def _cohort_ticks(index) -> list[str]:
    return [str(p) for p in index]


def save(fig: Figure, path) -> None:
    """Write a figure as PNG at >=150 dpi with a tight bounding box and an explicit background."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=t.SURFACE)


# --------------------------------------------------------------------------- 01 cleaning
def cleaning_waterfall(waterfall: pd.DataFrame) -> Figure:
    """Rows remaining after each cleaning rule (horizontal bars, raw at the top)."""
    w = waterfall.reset_index(drop=True)
    removed = w.iloc[1:]
    biggest = removed.loc[removed["PctOfRawRemoved"].idxmax()]
    others = removed.drop(biggest.name)["PctOfRawRemoved"].sum()
    fig, ax = _figure(10, 4.2)
    y = np.arange(len(w))[::-1]
    ax.barh(y, w["RowsRemaining"], height=0.58, color=t.PRIMARY)
    for yi, (_, r) in zip(y, w.iterrows()):
        label = f"{r['RowsRemaining']:,.0f}"
        if r["RowsRemoved"] > 0:
            label += f"   (-{r['RowsRemoved']:,.0f}, -{r['PctOfRawRemoved']:.2%} of raw)"
        ax.text(r["RowsRemaining"] + w["RowsRemaining"].max() * 0.012, yi, label,
                va="center", ha="left", fontsize=9, color=t.INK)
    ax.set_yticks(y, w["Step"])
    ax.set_xlim(0, w["RowsRemaining"].max() * 1.42)
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("Line-item rows remaining")
    ax.grid(axis="y", visible=False)
    _titles(ax,
            f"Missing customer IDs remove {biggest['PctOfRawRemoved']:.1%} of rows; "
            f"every other rule together removes {others:.1%}",
            f"{w['RowsRemaining'].iloc[-1]:,.0f} of {w['RowsRemaining'].iloc[0]:,.0f} raw rows "
            f"({w['PctOfRawRemaining'].iloc[-1]:.1%}) are kept for the analysis.")
    return fig


def orders_per_customer_hist(customers: pd.DataFrame, cap: int = 30) -> Figure:
    """Distribution of orders per customer; everything at or above `cap` is pooled."""
    n = customers["Orders"].clip(upper=cap).value_counts().sort_index()
    one = float((customers["Orders"] == 1).mean())
    fig, ax = _figure(9, 4)
    ax.bar(n.index, n.values, width=0.8, color=t.PRIMARY)
    ax.set_xlabel(f"Orders per customer (last bar = {cap}+)")
    ax.set_ylabel("Customers")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="x", visible=False)
    _titles(ax, f"{one:.0%} of customers ordered exactly once; the median customer placed "
                f"{customers['Orders'].median():.0f} orders",
            f"Mean {customers['Orders'].mean():.1f} orders vs median "
            f"{customers['Orders'].median():.0f} - a long wholesale tail pulls the mean up.")
    return fig


def order_value_hist(orders: pd.DataFrame, bins: int = 60) -> Figure:
    """Distribution of order value on a log x-axis."""
    v = orders["Revenue"]
    edges = np.logspace(np.log10(max(v.min(), 0.5)), np.log10(v.max()), bins)
    fig, ax = _figure(9, 4)
    ax.hist(v, bins=edges, color=t.PRIMARY, edgecolor=t.SURFACE, linewidth=0.6)
    ax.set_xscale("log")
    ax.axvline(v.median(), color=t.INK, linewidth=1.2, linestyle="--")
    ax.annotate(f"median £{v.median():,.0f}", xy=(v.median(), 1), xycoords=("data", "axes fraction"),
                xytext=(6, -12), textcoords="offset points", fontsize=9, color=t.INK)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x:,.0f}"))
    ax.set_xlabel("Order value (£, log scale)")
    ax.set_ylabel("Orders")
    ax.yaxis.set_major_formatter(THOUSANDS)
    _titles(ax, f"A typical order is worth £{v.median():,.0f}, but the mean is £{v.mean():,.0f} - "
                "order value is heavily right-skewed",
            f"{len(v):,} orders; the largest is £{v.max():,.0f}. Medians, not means, describe typical behaviour.")
    return fig


def top_countries_bar(countries: pd.DataFrame) -> Figure:
    """Top countries by customers (country of first order)."""
    c = countries.iloc[::-1]
    fig, ax = _figure(9, 4.2)
    ax.barh(c.index, c["Customers"], height=0.6, color=t.PRIMARY)
    for i, (_, r) in enumerate(c.iterrows()):
        ax.text(r["Customers"] + c["Customers"].max() * 0.01, i,
                f"{r['Customers']:,.0f}  ({r['CustomerShare']:.1%})", va="center", fontsize=9, color=t.INK)
    ax.set_xlim(0, c["Customers"].max() * 1.18)
    ax.set_xlabel("Customers")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="y", visible=False)
    top = countries.iloc[0]
    _titles(ax, f"{top['CustomerShare']:.0%} of identified customers are in {top.name}",
            "Country of each customer's first order; top 10 shown.")
    return fig


def monthly_orders_line(monthly: pd.DataFrame, last_complete_month) -> Figure:
    """Orders per calendar month; the partial final month is drawn as a hollow marker."""
    m = monthly.copy()
    complete = m[m.index <= last_complete_month]
    partial = m[m.index > last_complete_month]
    x = np.arange(len(m))
    peak = complete["Orders"].idxmax()
    fig, ax = _figure(10, 4)
    ax.plot(x[:len(complete)], complete["Orders"], color=t.PRIMARY, marker="o")
    if len(partial):
        ax.plot(x[len(complete) - 1:], [complete["Orders"].iloc[-1], *partial["Orders"]],
                color=t.DEEMPHASIS, linestyle=":", linewidth=1.5)
        ax.scatter(x[len(complete):], partial["Orders"], s=40, facecolor=t.SURFACE,
                   edgecolor=t.DEEMPHASIS, linewidth=1.5, zorder=3)
        ax.annotate("partial month", xy=(x[-1], partial["Orders"].iloc[-1]), xytext=(-6, -14),
                    textcoords="offset points", ha="right", fontsize=9, color=t.MUTED)
    step = max(1, len(m) // 12)
    ax.set_xticks(x[::step], [str(p) for p in m.index[::step]], rotation=45, ha="right")
    ax.set_ylabel("Orders")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(THOUSANDS)
    _titles(ax, f"Order volume is seasonal - {_month_label(peak)} is the busiest month "
                f"({complete.loc[peak, 'Orders']:,.0f} orders)",
            "Distinct invoices per calendar month after cleaning.")
    return fig


# --------------------------------------------------------------------------- 02 heatmap
def retention_heatmap(pct: pd.DataFrame, sizes: pd.Series, first_month, title: str | None = None,
                      subtitle: str | None = None) -> Figure:
    """Cohort x month-index retention heatmap.

    Month 0 is 100% by definition, so it is drawn in neutral grey and left out of the colour
    scale; unobserved cells (NaN) are blank. The first cohort row carries an explicit label
    because it is left-censored.
    """
    p = pct.dropna(axis=1, how="all")
    sizes = sizes.astype(int)
    body = p.drop(columns=0) if 0 in p.columns else p
    vmax = float(np.ceil(np.nanmax(body.to_numpy()) * 20) / 20)
    true_new = p.loc[p.index > first_month]
    m1 = float(true_new[1].mean()) if 1 in true_new else float("nan")

    fig, ax = _figure(0.56 * len(p.columns) + 3.6, 0.36 * len(p) + 1.9)
    data = p.to_numpy(dtype=float).copy()
    shown = np.ma.masked_invalid(data)
    if 0 in p.columns:
        shown[:, list(p.columns).index(0)] = np.ma.masked
    im = ax.imshow(shown, cmap=SEQ_CMAP, vmin=0, vmax=vmax, aspect="auto")
    for i in range(data.shape[0]):
        for j, ci in enumerate(p.columns):
            v = data[i, j]
            if np.isnan(v):
                continue
            if ci == 0:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="#eeede8", edgecolor="none"))
                ax.text(j, i, "100", ha="center", va="center", fontsize=7.5, color=t.MUTED)
            else:
                ax.text(j, i, f"{v * 100:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > vmax * 0.5 else t.INK)
    # hairline surface gaps between cells
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color=t.SURFACE, linewidth=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(range(len(p.columns)), [str(c) for c in p.columns])
    ylabels = [f"{_month_label(cm)} - includes pre-existing customers  (n={sizes[cm]:,})"
               if cm == first_month else f"{cm}  (n={sizes[cm]:,})" for cm in p.index]
    ax.set_yticks(range(len(p.index)), ylabels)
    ax.set_xlabel("Months since first order (cohort index)")
    ax.set_ylabel("Cohort (month of first order)")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("Share of cohort active in month (%)", color=t.INK_SECONDARY)
    cb.ax.yaxis.set_major_formatter(PCT)
    cb.outline.set_visible(False)
    _titles(ax, title or (f"Only about {m1:.0%} of a new cohort orders again the following month - "
                          "and the rate barely decays after that"),
            subtitle or ("Cell = % of the cohort placing at least one order that month. Month 0 is 100% by "
                         "definition (grey). Blank cells are months not yet fully observed, not zeros."),
            wrap=110)
    return fig


# --------------------------------------------------------------------------- 03 curves
def retention_curves(pct: pd.DataFrame, cohorts: list, first_month=None) -> Figure:
    """Retention curves (month 1 onwards) for a handful of selected cohorts (max 6)."""
    cohorts = list(cohorts)[:6]
    sub = pct.loc[cohorts].drop(columns=0, errors="ignore")
    hi = float(np.nanmax(sub.to_numpy()))
    # like-for-like comparison: only the months every selected cohort has fully observed
    common = sub.dropna(axis=1, how="any")
    early = common.mean(axis=1)
    best, worst = early.idxmax(), early.idxmin()
    fig, ax = _figure(10, 4.6)
    for color, cm in zip(t.SERIES, cohorts):
        s = sub.loc[cm].dropna()
        note = " (pre-existing customers)" if first_month is not None and cm == first_month else ""
        ax.plot(s.index, s.values, color=color, marker="o", markeredgecolor=t.SURFACE,
                markeredgewidth=1.2, label=f"{_month_label(cm)}{note}")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, max(0.05, hi * 1.15))
    ax.set_xlim(0.5, sub.columns.max() + 0.5)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Months since first order (cohort index)")
    ax.set_ylabel("Share of cohort active (%)")
    ax.legend(title="Cohort", ncols=min(len(cohorts), 3), loc="upper center",
              bbox_to_anchor=(0.5, -0.17), title_fontsize=9)
    _titles(ax, f"Over their first {common.shape[1]} months the {_month_label(best)} cohort kept "
                f"{early[best]:.0%} of customers active per month; the {_month_label(worst)} cohort only "
                f"{early[worst]:.0%}",
            "Month 0 (100% by definition) is omitted so the differences are visible. Each line stops at the "
            "last fully observed month; single cohorts are noisy, so read the level rather than each wiggle.")
    return fig


# --------------------------------------------------------------------------- 04 average curve
def avg_retention_curve(avg: pd.DataFrame) -> Figure:
    """Average retention by month index (true-new cohorts) with the number of cohorts behind
    each point shown in a separate panel beneath."""
    a = avg.reset_index()
    change = a["mean_retention"].diff()
    worst = int(change.idxmin())
    m1 = float(a.loc[a["month_index"] == 1, "mean_retention"].iloc[0])
    later = a.loc[a["month_index"] >= 1, "mean_retention"]
    fig, (ax, ax2) = _figure(10, 5.6, nrows=2, sharex=True, height_ratios=[3.4, 1])
    ax.plot(a["month_index"], a["mean_retention"], color=t.PRIMARY, marker="o",
            markeredgecolor=t.SURFACE, markeredgewidth=1.2)
    x0, x1 = a.loc[worst - 1, "month_index"], a.loc[worst, "month_index"]
    y1 = a.loc[worst, "mean_retention"]
    ax.annotate(f"{change[worst] * 100:+.1f} pp between month {x0} and month {x1}",
                xy=(x1, y1), xytext=(x1 + 1.6, y1 + 0.42), fontsize=10, color=t.INK,
                arrowprops={"arrowstyle": "-", "color": t.MUTED, "linewidth": 0.8})
    if len(a) > 2 and change.iloc[2] > 0:
        ax.annotate(f"month 2: {a.loc[2, 'mean_retention']:.1%} (slight rebound)",
                    xy=(a.loc[2, "month_index"], a.loc[2, "mean_retention"]),
                    xytext=(a.loc[2, "month_index"] + 1.6, a.loc[2, "mean_retention"] + 0.17),
                    fontsize=9, color=t.INK_SECONDARY,
                    arrowprops={"arrowstyle": "-", "color": t.MUTED, "linewidth": 0.8})
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean share of cohort active (%)")
    ax2.bar(a["month_index"], a["n_cohorts"], width=0.6, color=t.DEEMPHASIS)
    ax2.set_ylabel("Cohorts\naveraged")
    ax2.set_xlabel("Months since first order (cohort index)")
    ax2.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax2.grid(axis="x", visible=False)
    _titles(ax, f"{1 - m1:.0%} of new customers are not active the month after their first order; "
                f"retention then holds between {later.min():.0%} and {later.max():.0%}",
            "Mean across true-new cohorts (the left-censored first cohort is excluded), using only cohorts "
            "that have fully observed each month. Later points rest on fewer cohorts - see the lower panel.")
    return fig


# --------------------------------------------------------------------------- 05 repeat within window
def repeat_window_by_cohort(summary: pd.DataFrame, first_month, window: int = 90) -> Figure:
    """Repeat-within-window by cohort, fully observed cohorts only, with a dashed mean line."""
    col = f"Repeat{window}d"
    s = summary.loc[summary["FullyObserved"], col]
    mean_all = float(s.mean())
    true_new = s[s.index > first_month]
    worst, best = true_new.idxmin(), true_new.idxmax()
    fig, ax = _figure(11, 4.6)
    x = np.arange(len(s))
    colors = [t.DEEMPHASIS if cm == first_month else t.PRIMARY for cm in s.index]
    ax.bar(x, s.values, width=0.72, color=colors)
    ax.axhline(mean_all, color=t.INK, linestyle="--", linewidth=1.2,
               label=f"Mean of fully observed cohorts: {mean_all:.1%}")
    ax.legend(loc="upper right")
    for cm in {worst, best, s.index[0]}:
        i = list(s.index).index(cm)
        ax.text(i, s[cm] + 0.012, f"{s[cm]:.1%}", ha="center", fontsize=9, color=t.INK)
    if s.index[0] == first_month:
        ax.annotate("includes pre-existing\ncustomers", xy=(0, s.iloc[0] + 0.045), xytext=(0.9, s.max() + 0.11),
                    fontsize=8.5, color=t.MUTED, ha="left", va="center",
                    arrowprops={"arrowstyle": "-", "color": t.MUTED, "linewidth": 0.8})
    ax.set_xticks(x, _cohort_ticks(s.index), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, s.max() + 0.17)
    ax.set_ylabel(f"Customers with a 2nd order within {window} days (%)")
    ax.set_xlabel("Cohort (month of first order)")
    ax.grid(axis="x", visible=False)
    _titles(ax, f"Like-for-like, {true_new.min():.0%} to {true_new.max():.0%} of a new cohort re-orders "
                f"within {window} days - {_month_label(worst)} is the weakest cohort",
            f"Only cohorts whose full {window}-day window lies inside the data ({len(s)} of {len(summary)}). "
            f"True-new cohorts alone average {true_new.mean():.1%}.")
    return fig


def repeat_ever_vs_window(summary: pd.DataFrame, window: int = 90) -> Figure:
    """Repeat-ever next to repeat-within-window per cohort, to show the observation-time bias."""
    col = f"Repeat{window}d"
    fig, ax = _figure(11, 4.6)
    x = np.arange(len(summary))
    ax.plot(x, summary["RepeatEver"], color=t.ACCENT, marker="o", markeredgecolor=t.SURFACE,
            markeredgewidth=1.2, label="Repeat-ever (biased by observation time)")
    ax.plot(x, summary[col], color=t.PRIMARY, marker="o", markeredgecolor=t.SURFACE,
            markeredgewidth=1.2, label=f"Repeat-within-{window}-days (like-for-like)")
    ax.set_xticks(x, _cohort_ticks(summary.index), rotation=45, ha="right")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of cohort (%)")
    ax.set_xlabel("Cohort (month of first order)")
    ax.legend(loc="lower left")
    _titles(ax, f"Repeat-ever falls from {summary['RepeatEver'].iloc[0]:.0%} to "
                f"{summary['RepeatEver'].iloc[-1]:.0%} simply because newer cohorts have been watched for "
                "less time",
            f"The {window}-day metric gives every cohort the same opportunity; it is blank for cohorts "
            "whose window is not fully observed.")
    return fig


# --------------------------------------------------------------------------- 06 survival
def survival_to_second(km: pd.DataFrame, readings: dict) -> Figure:
    """Kaplan-Meier 'share who have placed a 2nd order by day N' with fixed-horizon readings."""
    fig, ax = _figure(10, 4.8)
    ax.step(km["day"], km["repeated_by"], where="post", color=t.PRIMARY)
    for d, v in readings.items():
        if v != v:          # NaN: horizon beyond the curve
            continue
        d = int(d)
        ax.scatter([d], [v], s=42, color=t.PRIMARY, edgecolor=t.SURFACE, linewidth=1.5, zorder=3)
        ax.annotate(f"day {d}: {v:.1%}", xy=(d, v), xytext=(8, -14), textcoords="offset points",
                    fontsize=9.5, color=t.INK)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, 1)
    ax.set_xlim(-4, km["day"].max() + 42)
    ax.set_xlabel("Days since first order")
    ax.set_ylabel("Customers who have placed a 2nd order (%)")
    r = {int(k): v for k, v in readings.items() if v == v}
    mid, last = (90 if 90 in r else sorted(r)[len(r) // 2]), max(r)
    _titles(ax, f"{r[mid]:.0%} of customers place a second order within {mid} days, and "
                f"{r[last]:.0%} within {last} days",
            "Kaplan-Meier estimate over all identified customers; customers who have not yet re-ordered are "
            "censored at the end of the data rather than counted as lost. The jump at day 0 is same-day "
            "second invoices.")
    return fig


# --------------------------------------------------------------------------- 07 histogram
def days_to_second_hist(hist: pd.DataFrame, median_days: float, same_day_share: float) -> Figure:
    """Days between first and second order (repeat customers), weekly bins, capped."""
    h = hist.reset_index(drop=True)
    day0, body, tail = h.iloc[0], h.iloc[1:-1], h.iloc[-1]
    cap = int(tail["bin_start"])
    fig, ax = _figure(10, 4.6)
    width = body["bin_end"] - body["bin_start"] + 1
    ax.bar(body["bin_start"], body["customers"], width=width - 1.2, align="edge", color=t.PRIMARY,
           label="Weekly bins (days 1 onwards)")
    ax.bar([-4.5], [day0["customers"]], width=4, align="edge", color=t.ACCENT,
           label="Day 0 (same-day second invoice)")
    ax.bar([cap + 6], [tail["customers"]], width=width.iloc[0] - 1.2, align="edge", color=t.DEEMPHASIS,
           label=f"{cap}+ days (pooled)")
    top = max(h["customers"])
    ax.axvline(median_days, color=t.INK, linestyle="--", linewidth=1.2, ymax=0.8)
    ax.annotate(f"median {median_days:.0f} days", xy=(median_days, 0.74), xycoords=("data", "axes fraction"),
                xytext=(6, 0), textcoords="offset points", fontsize=9.5, color=t.INK)
    ax.annotate(f"day 0: {day0['customers']:,.0f} customers ({same_day_share:.1%}) - often split or "
                "corrected orders", xy=(-2.5, day0["customers"]), xytext=(-2.5, top * 1.13),
                fontsize=9, color=t.INK, va="center", ha="left",
                arrowprops={"arrowstyle": "-", "color": t.MUTED, "linewidth": 0.8})
    ax.set_xlim(-12, cap + 22)
    ax.set_ylim(0, top * 1.22)
    ax.set_xlabel("Days between first and second order")
    ax.set_ylabel("Repeat customers")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    _titles(ax, f"Half of repeat customers come back within {median_days:.0f} days; "
                f"{same_day_share:.1%} place their 'second order' on the same day",
            "Customers with at least two orders. The day-0 bar is a single day, the blue bars are 7-day bins.")
    return fig


# --------------------------------------------------------------------------- 08 revenue
def revenue_split(split: pd.DataFrame, first_month=None) -> Figure:
    """First vs repeat revenue per cohort (stacked bars) and the repeat share (separate panel).

    The left-censored first cohort dwarfs every other bar, so when `first_month` is given its bar
    is faded, runs off the top of the axis and is labelled with its true total instead.
    """
    s = split
    share = float(s["Repeat"].sum() / s["Total"].sum())
    fig, (ax, ax2) = _figure(11, 6.2, nrows=2, sharex=True, height_ratios=[2.6, 1.3])
    x = np.arange(len(s))
    alphas = [0.45 if cm == first_month else 1.0 for cm in s.index]
    for xi, (cm, r), a in zip(x, s.iterrows(), alphas):
        ax.bar(xi, r["First"] / 1e3, width=0.72, color=t.PRIMARY, alpha=a, edgecolor=t.SURFACE,
               linewidth=1.0, label="First orders" if xi == len(s) - 1 else None)
        ax.bar(xi, r["Repeat"] / 1e3, bottom=r["First"] / 1e3, width=0.72, color=t.ACCENT, alpha=a,
               edgecolor=t.SURFACE, linewidth=1.0,
               label="Repeat orders (2nd onwards)" if xi == len(s) - 1 else None)
    others = s.loc[s.index != first_month, "Total"] if first_month is not None else s["Total"]
    if first_month in s.index and s.loc[first_month, "Total"] > 1.5 * others.max():
        ax.set_ylim(0, others.max() / 1e3 * 1.45)
        ax.annotate(f"{_month_label(first_month)}: £{s.loc[first_month, 'Total'] / 1e6:.1f}M in total (bar runs "
                    "off the scale)\nincludes pre-existing customers",
                    xy=(0.45, others.max() / 1e3 * 1.3), xytext=(1.3, others.max() / 1e3 * 1.3),
                    fontsize=9, color=t.INK, va="center",
                    arrowprops={"arrowstyle": "-", "color": t.MUTED, "linewidth": 0.8})
    ax.set_ylabel("Revenue to date (£ thousand)")
    ax.yaxis.set_major_formatter(THOUSANDS)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    ax2.plot(x, s["RepeatShare"], color=t.ACCENT, marker="o", markeredgecolor=t.SURFACE, markeredgewidth=1.2)
    ax2.yaxis.set_major_formatter(PCT)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Repeat share of\ncohort revenue (%)")
    ax2.set_xticks(x, _cohort_ticks(s.index), rotation=45, ha="right")
    ax2.set_xlabel("Cohort (month of first order)")
    for i in (0, len(s) - 1):
        ax2.annotate(f"{s['RepeatShare'].iloc[i]:.0%}", xy=(x[i], s["RepeatShare"].iloc[i]), xytext=(0, 7),
                     textcoords="offset points", ha="center", fontsize=9, color=t.INK)
    _titles(ax, f"Repeat orders bring in {share:.0%} of all revenue - first orders are a small "
                "slice of every mature cohort",
            "Revenue each cohort has generated so far, gross of returns. Newer cohorts have had less time, "
            "so their bars and repeat shares are lower by construction.")
    return fig


def cumulative_revenue_curves(cum: pd.DataFrame, cohorts: list) -> Figure:
    """Cumulative revenue per cohort customer by months since first order (max 6 cohorts)."""
    cohorts = list(cohorts)[:6]
    fig, ax = _figure(10, 4.6)
    for color, cm in zip(t.SERIES, cohorts):
        s = cum.loc[cm].dropna()
        ax.plot(s.index, s.values, color=color, label=_month_label(cm))
    first = cum.loc[cohorts[0]].dropna()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"£{v:,.0f}"))
    ax.set_ylim(0, None)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Months since first order (cohort index)")
    ax.set_ylabel("Cumulative revenue per cohort customer (£, mean)")
    ax.legend(title="Cohort", loc="upper left", title_fontsize=9)
    _titles(ax, f"A customer's value keeps compounding: the {_month_label(cohorts[0])} cohort went from "
                f"£{first.iloc[0]:,.0f} per customer in month 0 to £{first.iloc[-1]:,.0f} by month "
                f"{first.index[-1]}",
            "Mean revenue per customer in the cohort (repeaters and non-repeaters alike). Means are pulled "
            "up by wholesale buyers; read the shape, not the level.")
    return fig


# --------------------------------------------------------------------------- 09 / 10 segments
def segment_repeat_rates(segments: pd.DataFrame, window: int = 90) -> Figure:
    """Repeat-within-window for every segment value, grouped by segment type, with sample sizes."""
    col = f"repeat_{window}d"
    groups = [(k, g) for k, g in segments.groupby("segment_type", sort=False)]
    spread = {k: (g[col].max() - g[col].min()) * 100 for k, g in groups}
    widest, narrowest = max(spread, key=spread.get), min(spread, key=spread.get)
    n_rows = len(segments) + len(groups)
    fig, ax = _figure(10, 0.42 * n_rows + 1.6)
    y, ticks, labels = 0, [], []
    for k, g in groups:
        ax.text(0, y, t.SEGMENT_LABELS.get(k, k), fontsize=10, fontweight="bold", color=t.INK, va="center")
        y += 1
        for _, r in g.iterrows():
            ax.barh(y, r[col], height=0.6, color=t.PRIMARY)
            flag = "  - small sample" if r["small_sample"] else ""
            ax.text(r[col] + 0.008, y, f"{r[col]:.1%}   n={r['n']:,}{flag}", va="center", fontsize=9, color=t.INK)
            ticks.append(y)
            labels.append(t.segment_value_label(k, r["segment_value"]))
            y += 1
        y += 0.4
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.6, -0.8)
    ax.set_xlim(0, segments[col].max() * 1.45)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_xlabel(f"Customers with a 2nd order within {window} days (%)")
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _titles(ax, f"{t.SEGMENT_LABELS.get(widest, widest)} separates repeaters most "
                f"({spread[widest]:.1f} pp spread); {t.SEGMENT_LABELS.get(narrowest, narrowest).lower()} "
                f"barely matters ({spread[narrowest]:.1f} pp)",
            "True-new customers in fully observed cohorts only, so every customer had the same "
            f"{window} days to return.")
    return fig


def first_order_tier(segments: pd.DataFrame, tiers: pd.DataFrame, window: int = 90) -> Figure:
    """Repeat-within-window by first-order value tier (ordered ramp, sample sizes on the bars)."""
    col = f"repeat_{window}d"
    g = segments[segments["segment_type"] == "FirstOrderTier"].set_index("segment_value")
    order = [k for k in ("Low", "Mid", "High") if k in g.index]
    g = g.loc[order]
    fig, ax = _figure(7.5, 4.6)
    x = np.arange(len(g))
    ax.bar(x, g[col], width=0.58, color=t.ORDINAL3[:len(g)])
    for i, (k, r) in enumerate(g.iterrows()):
        ax.text(i, r[col] + 0.012, f"{r[col]:.1%}", ha="center", fontsize=11, fontweight="bold", color=t.INK)
        ax.text(i, r[col] + 0.05, f"n={r['n']:,}", ha="center", fontsize=9, color=t.INK_SECONDARY)
    ticks = [f"{k}\n£{tiers.loc[k, 'min']:,.0f} - £{tiers.loc[k, 'max']:,.0f}" for k in order]
    ax.set_xticks(x, ticks)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, g[col].max() + 0.14)
    ax.set_ylabel(f"Customers with a 2nd order within {window} days (%)")
    ax.set_xlabel("Value of the customer's first order (terciles)")
    ax.grid(axis="x", visible=False)
    _titles(ax, "Bigger first orders, more second orders: "
                + " -> ".join(f"{g.loc[k, col]:.0%}" for k in order),
            "True-new customers in fully observed cohorts. A correlation, not proof that bigger baskets "
            "cause loyalty - large first orders also mark wholesale buyers.", wrap=64)
    return fig


def segment_retention_curves(curves: dict[str, pd.Series], title: str, subtitle: str | None = None) -> Figure:
    """Average retention curves (month 1 onwards) for up to six segments on one axis."""
    fig, ax = _figure(10, 4.6)
    hi = 0.0
    for color, (name, s) in zip(t.SERIES, list(curves.items())[:6]):
        s = s[s.index >= 1]
        hi = max(hi, float(s.max()))
        ax.plot(s.index, s.values, color=color, marker="o", markeredgecolor=t.SURFACE,
                markeredgewidth=1.2, label=name)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_ylim(0, hi * 1.2)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_xlabel("Months since first order (cohort index)")
    ax.set_ylabel("Mean share of cohort active (%)")
    ax.legend(ncols=min(len(curves), 3), loc="upper center", bbox_to_anchor=(0.5, -0.17))
    _titles(ax, title, subtitle)
    return fig
