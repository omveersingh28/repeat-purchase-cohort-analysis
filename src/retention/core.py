"""src/retention/core.py — cleaning, orders, cohorts, retention, repeat and survival metrics.

Pure functions, no file I/O except `load_raw`. No dates are hard-coded: every boundary is
derived from the data so the pipeline still works if the dataset is extended or swapped.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

#: Stock codes that identify a real product. In Online Retail II every purely numeric code is
#: 5 digits, optionally followed by 1-2 letters (e.g. "22041", "79323P"). Everything else is
#: postage, fees, samples, gift vouchers, manual adjustments or test rows. Overridable.
PRODUCT_CODE_RE = re.compile(r"^\d{5}[A-Za-z]{0,2}$")

REQUIRED_COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"]


# --------------------------------------------------------------------------- loading & cleaning
def load_raw(path) -> pd.DataFrame:
    """Read the raw dataset (.csv / .xlsx) into one DataFrame with canonical column names."""
    path = str(path)
    if path.lower().endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(path, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        df = pd.concat(sheets.values(), ignore_index=True)
    else:
        df = pd.read_csv(path, dtype={"Invoice": str, "StockCode": str})
    df = df.rename(columns={"Customer ID": "CustomerID", "UnitPrice": "Price",
                            "InvoiceNo": "Invoice", "CustomerID ": "CustomerID"})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing expected columns: {missing}; got {list(df.columns)}")
    return df


def clean(df: pd.DataFrame, product_re: re.Pattern = PRODUCT_CODE_RE,
          drop_non_product: bool = True) -> tuple[pd.DataFrame, dict]:
    """Apply the documented cleaning rules. Returns (line_items, waterfall_log).

    Rules, in order: drop rows with no CustomerID -> drop cancellations ('C') and
    adjustments ('A') -> drop Quantity<=0 or Price<=0 -> drop non-product stock codes.
    """
    log: dict = {"raw_rows": int(len(df))}
    out = df.dropna(subset=["CustomerID"])
    log["after_missing_customer"] = int(len(out))

    inv = out["Invoice"].astype(str).str.upper()
    out = out[~inv.str.startswith(("C", "A"))]
    log["after_cancel_adjust"] = int(len(out))

    out = out[(out["Quantity"] > 0) & (out["Price"] > 0)]
    log["after_qty_price"] = int(len(out))

    is_product = out["StockCode"].astype(str).str.match(product_re)
    log["non_product_rows"] = int((~is_product).sum())
    log["non_product_codes"] = sorted(out.loc[~is_product, "StockCode"].astype(str).unique().tolist())
    rev_all = float((out["Quantity"] * out["Price"]).sum())
    rev_np = float((out.loc[~is_product, "Quantity"] * out.loc[~is_product, "Price"]).sum())
    log["non_product_revenue_share"] = round(rev_np / rev_all, 6) if rev_all else 0.0
    if drop_non_product:
        out = out[is_product]
    log["after_non_product"] = int(len(out))

    out = out.copy()
    out["CustomerID"] = out["CustomerID"].astype(int)
    out["Revenue"] = out["Quantity"] * out["Price"]
    log["customers"] = int(out["CustomerID"].nunique())
    log["invoices"] = int(out["Invoice"].nunique())
    log["revenue"] = float(out["Revenue"].sum())
    log["date_min"] = str(out["InvoiceDate"].min())
    log["date_max"] = str(out["InvoiceDate"].max())
    return out, log


# --------------------------------------------------------------------------- time boundaries
def period_bounds(lines: pd.DataFrame) -> dict:
    """Derive every time boundary from the data — nothing is hard-coded.

    first_month        : month of the earliest transaction (its cohort is left-censored)
    last_complete_month: latest month fully covered by the data (partial final month excluded)
    data_end           : timestamp of the last transaction
    """
    data_end = lines["InvoiceDate"].max()
    first_month = lines["InvoiceDate"].min().to_period("M")
    last_month = data_end.to_period("M")
    last_complete_month = last_month if data_end >= last_month.end_time.normalize() else last_month - 1
    return {"data_end": data_end, "first_month": first_month,
            "last_complete_month": last_complete_month, "last_month": last_month}


# --------------------------------------------------------------------------- orders & customers
def build_orders(lines: pd.DataFrame) -> pd.DataFrame:
    """Collapse line items to one row per order (invoice), with cohort and sequence columns."""
    o = (lines.groupby(["CustomerID", "Invoice"], as_index=False)
         .agg(OrderDate=("InvoiceDate", "min"), Revenue=("Revenue", "sum"),
              Items=("Quantity", "sum"), Country=("Country", "first")))
    o = o.sort_values(["CustomerID", "OrderDate", "Invoice"], kind="mergesort").reset_index(drop=True)
    o["OrderNumber"] = o.groupby("CustomerID").cumcount() + 1
    o["OrderMonth"] = o["OrderDate"].dt.to_period("M")
    first_month = o.loc[o["OrderNumber"] == 1].set_index("CustomerID")["OrderMonth"]
    o["CohortMonth"] = o["CustomerID"].map(first_month)
    o["CohortIndex"] = (o["OrderMonth"].astype("period[M]") - o["CohortMonth"].astype("period[M]")
                        ).apply(lambda x: x.n)
    return o


def build_customers(orders: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """One row per customer: cohort, order counts, revenue, time-to-second-order and segments."""
    data_end, first_month = bounds["data_end"], bounds["first_month"]
    g = orders.groupby("CustomerID")
    c = g.agg(CohortMonth=("CohortMonth", "first"), FirstOrderDate=("OrderDate", "min"),
              LastOrderDate=("OrderDate", "max"), Orders=("Invoice", "nunique"),
              TotalRevenue=("Revenue", "sum"))
    firsts = orders.loc[orders["OrderNumber"] == 1].set_index("CustomerID")
    c["FirstOrderValue"] = firsts["Revenue"]
    c["Country"] = firsts["Country"]
    seconds = orders.loc[orders["OrderNumber"] == 2].set_index("CustomerID")["OrderDate"]
    c["SecondOrderDate"] = seconds
    c["DaysToSecond"] = (c["SecondOrderDate"].dt.normalize() - c["FirstOrderDate"].dt.normalize()).dt.days
    c["IsRepeat"] = c["Orders"] >= 2
    c["ObservedDays"] = (data_end.normalize() - c["FirstOrderDate"].dt.normalize()).dt.days

    # Segments
    c["IsTrueNew"] = c["CohortMonth"] > first_month          # False = left-censored first cohort
    c["Region"] = np.where(c["Country"].eq("United Kingdom"), "UK", "International")
    c["IsQ4Cohort"] = c["CohortMonth"].apply(lambda p: p.month in (10, 11, 12))
    c["CohortYear"] = c["CohortMonth"].apply(lambda p: p.year)
    try:
        c["FirstOrderTier"] = pd.qcut(c["FirstOrderValue"], 3, labels=["Low", "Mid", "High"])
    except ValueError:                                        # too few distinct values (tiny test data)
        c["FirstOrderTier"] = pd.Categorical(["Mid"] * len(c), categories=["Low", "Mid", "High"])
    return c


# --------------------------------------------------------------------------- retention
def retention_matrix(orders: pd.DataFrame, bounds: dict,
                     customer_ids: pd.Index | None = None) -> dict:
    """Cohort x month-index retention. Cells whose calendar month is not fully observed are NaN.

    Pass `customer_ids` to compute the matrix for a subset (a segment) while keeping each
    customer in their original cohort.
    """
    o = orders if customer_ids is None else orders[orders["CustomerID"].isin(customer_ids)]
    last_complete = bounds["last_complete_month"]
    counts = o.groupby(["CohortMonth", "CohortIndex"])["CustomerID"].nunique().unstack(1).sort_index()
    counts = counts.reindex(columns=sorted(counts.columns))
    sizes = counts[0].copy()
    pct = counts.divide(sizes, axis=0)

    mask = pd.DataFrame(False, index=counts.index, columns=counts.columns)
    for cm in counts.index:
        for ci in counts.columns:
            if cm + ci > last_complete:
                mask.loc[cm, ci] = True
    counts = counts.mask(mask)
    pct = pct.mask(mask)
    keep = counts.index <= last_complete
    return {"counts": counts.loc[keep], "pct": pct.loc[keep], "sizes": sizes.loc[keep]}


def average_retention_curve(pct: pd.DataFrame, exclude_first_cohort: bool = True,
                            first_month: pd.Period | None = None, min_cohorts: int = 3) -> pd.Series:
    """Mean retention by month index across cohorts that actually observed that month.

    Month N is averaged only over cohorts with a non-null value at N, so the curve is not
    dragged up by old cohorts alone. Indices backed by fewer than `min_cohorts` are dropped.
    """
    p = pct
    if exclude_first_cohort and first_month is not None:
        p = p.loc[p.index > first_month]
    n = p.notna().sum()
    return p.mean(skipna=True).where(n >= min_cohorts).dropna()


def dropoff_table(avg_curve: pd.Series) -> pd.DataFrame:
    """Month-over-month change in the average retention curve; the most negative row is the
    biggest drop-off point."""
    d = pd.DataFrame({"Retention": avg_curve})
    d["Change"] = d["Retention"].diff()
    d["PctOfPrevious"] = d["Retention"] / d["Retention"].shift(1)
    return d


# --------------------------------------------------------------------------- repeat behaviour
def cohort_summary(customers: pd.DataFrame, bounds: dict, window: int = 90) -> pd.DataFrame:
    """Per-cohort repeat metrics.

    RepeatEver is observation-time biased (older cohorts had longer to come back) and is kept
    only for context. Repeat{window}d is the like-for-like metric and is filled ONLY for cohorts
    whose last day plus the window is still inside the data, so every customer counted had the
    same opportunity to repeat.
    """
    data_end, last_complete = bounds["data_end"], bounds["last_complete_month"]
    c = customers[customers["CohortMonth"] <= last_complete]
    out = c.groupby("CohortMonth").agg(
        CohortSize=("Orders", "size"),
        RepeatEver=("IsRepeat", "mean"),
        AvgOrders=("Orders", "mean"),
        MedianFirstOrderValue=("FirstOrderValue", "median"),
        RevenuePerCustomer=("TotalRevenue", "mean"),
    )
    within = c["DaysToSecond"].notna() & (c["DaysToSecond"] <= window)
    rate = within.groupby(c["CohortMonth"]).mean()
    med = c.loc[within].groupby("CohortMonth")["DaysToSecond"].median()
    fully_observed = [cm for cm in out.index
                      if cm.end_time.normalize() + pd.Timedelta(days=window) <= data_end.normalize()]
    out[f"Repeat{window}d"] = rate.where(rate.index.isin(fully_observed))
    out[f"MedianDaysToSecond_{window}d"] = med.where(med.index.isin(fully_observed))
    out["FullyObserved"] = out.index.isin(fully_observed)
    return out


def survival_to_second(customers: pd.DataFrame, horizon: int = 365) -> pd.DataFrame:
    """Kaplan-Meier estimate of 'share who have placed a 2nd order by day t'.

    Customers who never repeated are right-censored at their observation window, so this is not
    biased by recent cohorts the way a raw percentage is. Returns day / at_risk / events /
    survival / repeated_by (= 1 - survival).
    """
    dur = np.where(customers["IsRepeat"], customers["DaysToSecond"], customers["ObservedDays"])
    dur = pd.Series(dur, index=customers.index).fillna(0).clip(lower=0).astype(int)
    event = customers["IsRepeat"].to_numpy().astype(bool)
    df = pd.DataFrame({"t": dur.to_numpy(), "e": event}).sort_values("t")

    rows, n_at_risk, s = [], len(df), 1.0
    for t, grp in df.groupby("t", sort=True):
        d = int(grp["e"].sum())
        if n_at_risk <= 0:
            break
        if d:
            s *= 1 - d / n_at_risk
        rows.append({"day": int(t), "at_risk": int(n_at_risk), "events": d,
                     "survival": s, "repeated_by": 1 - s})
        n_at_risk -= len(grp)
    km = pd.DataFrame(rows)
    return km[km["day"] <= horizon].reset_index(drop=True)


def repeat_by_day(km: pd.DataFrame, days=(30, 60, 90, 180, 365)) -> dict:
    """Read the KM curve at fixed horizons -> {'30': 0.21, ...} (NaN if beyond the curve)."""
    out = {}
    for d in days:
        sub = km[km["day"] <= d]
        out[str(d)] = float(sub["repeated_by"].iloc[-1]) if len(sub) else float("nan")
    return out


def revenue_split(orders: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Revenue per cohort from first orders vs repeat orders, plus the repeat share."""
    o = orders[orders["CohortMonth"] <= bounds["last_complete_month"]].copy()
    o["OrderType"] = np.where(o["OrderNumber"] == 1, "First", "Repeat")
    piv = o.pivot_table(index="CohortMonth", columns="OrderType", values="Revenue",
                        aggfunc="sum", fill_value=0.0)
    for col in ("First", "Repeat"):
        if col not in piv:
            piv[col] = 0.0
    piv["Total"] = piv["First"] + piv["Repeat"]
    piv["RepeatShare"] = np.where(piv["Total"] > 0, piv["Repeat"] / piv["Total"], np.nan)
    return piv[["First", "Repeat", "Total", "RepeatShare"]]


def segment_table(customers: pd.DataFrame, bounds: dict, by: str, window: int = 90) -> pd.DataFrame:
    """Repeat-within-window by segment, restricted to fully observed cohorts and true-new
    customers so the comparison is like-for-like. `n` is reported so small groups are visible."""
    data_end = bounds["data_end"]
    c = customers[customers["IsTrueNew"] & (customers["CohortMonth"] <= bounds["last_complete_month"])]
    ok = [cm for cm in c["CohortMonth"].unique()
          if cm.end_time.normalize() + pd.Timedelta(days=window) <= data_end.normalize()]
    c = c[c["CohortMonth"].isin(ok)]
    within = c["DaysToSecond"].notna() & (c["DaysToSecond"] <= window)
    out = pd.DataFrame({
        "n": c.groupby(by, observed=True).size(),
        f"Repeat{window}d": within.groupby(c[by], observed=True).mean(),
        "AvgOrders": c.groupby(by, observed=True)["Orders"].mean(),
        "MedianFirstOrderValue": c.groupby(by, observed=True)["FirstOrderValue"].median(),
        "RevenuePerCustomer": c.groupby(by, observed=True)["TotalRevenue"].mean(),
    })
    out["SmallSample"] = out["n"] < 100
    return out


# =========================================================================== extensions
# Everything below builds on the reference functions above. It adds presentation-ready tables
# and the headline-number assembly shared by the pipeline, the notebooks and the dashboard, so
# no notebook has to re-implement logic. None of it changes the semantics defined above.

SEGMENT_COLUMNS = ("Region", "FirstOrderTier", "IsQ4Cohort", "CohortYear")
KM_DAYS = (30, 60, 90, 180, 365)

#: Assumptions used ONLY in the illustrative impact estimates. They are not measured from data.
ILLUSTRATIVE_UPLIFT = 0.02      # +2 percentage points on a repeat / retention rate
GAP_CLOSED_SHARE = 0.25         # share of an observed segment gap an intervention might close
RETAINED_SHARE = 0.01           # 1 percentage point of the repeat-customer base kept from lapsing


def waterfall_table(log: dict) -> pd.DataFrame:
    """Row waterfall from the cleaning log: rows remaining and removed at each rule."""
    steps = [("Raw rows", "raw_rows"),
             ("Has a CustomerID", "after_missing_customer"),
             ("Not a cancellation / adjustment", "after_cancel_adjust"),
             ("Quantity > 0 and Price > 0", "after_qty_price"),
             ("Product stock code (final)", "after_non_product")]
    w = pd.DataFrame({"Step": [s for s, _ in steps], "RowsRemaining": [log[k] for _, k in steps]})
    w["RowsRemoved"] = (-w["RowsRemaining"].diff()).fillna(0).astype(int)
    w["PctOfRawRemoved"] = w["RowsRemoved"] / log["raw_rows"]
    w["PctOfPreviousRemoved"] = (w["RowsRemoved"] / w["RowsRemaining"].shift(1)).fillna(0.0)
    w["PctOfRawRemaining"] = w["RowsRemaining"] / log["raw_rows"]
    return w


def non_product_breakdown(lines_unfiltered: pd.DataFrame,
                          product_re: re.Pattern = PRODUCT_CODE_RE) -> pd.DataFrame:
    """Rows, revenue and customers per excluded stock code.

    `lines_unfiltered` is the output of `clean(..., drop_non_product=False)`.
    """
    is_product = lines_unfiltered["StockCode"].astype(str).str.match(product_re)
    npd = lines_unfiltered[~is_product]
    out = (npd.groupby("StockCode")
           .agg(Rows=("Invoice", "size"), Revenue=("Revenue", "sum"), Customers=("CustomerID", "nunique"))
           .sort_values(["Rows", "Revenue"], ascending=False))
    out["RevenueShare"] = out["Revenue"] / lines_unfiltered["Revenue"].sum()
    return out


def monthly_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Orders, active customers and revenue per calendar month."""
    return (orders.groupby("OrderMonth")
            .agg(Orders=("Invoice", "nunique"), Customers=("CustomerID", "nunique"),
                 Revenue=("Revenue", "sum")).sort_index())


def top_countries(customers: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Top-n countries by customers (country of the first order), with revenue and share."""
    t = (customers.groupby("Country")
         .agg(Customers=("Orders", "size"), Orders=("Orders", "sum"), Revenue=("TotalRevenue", "sum"))
         .sort_values(["Customers", "Revenue"], ascending=False))
    t["CustomerShare"] = t["Customers"] / len(customers)
    return t.head(n)


def average_retention_table(pct: pd.DataFrame, first_month: pd.Period | None = None,
                            min_cohorts: int = 3) -> pd.DataFrame:
    """`average_retention_curve` plus the number of cohorts backing each month index."""
    avg = average_retention_curve(pct, first_month=first_month, min_cohorts=min_cohorts)
    p = pct.loc[pct.index > first_month] if first_month is not None else pct
    n = p.notna().sum()
    return pd.DataFrame({"mean_retention": avg, "n_cohorts": n.reindex(avg.index).astype(int)}
                        ).rename_axis("month_index")


def select_cohorts(summary: pd.DataFrame, bounds: dict, window: int = 90) -> list:
    """Pick up to five cohorts for a readable curve chart, all derived from the data:
    earliest true-new, one six months later, the first Q4 cohort, the largest fully observed
    cohort of the final calendar year, and the latest fully observed cohort."""
    first = bounds["first_month"]
    true_new = summary[summary.index > first]
    full = true_new[true_new["FullyObserved"]]
    picks = []
    if len(true_new):
        picks.append(true_new.index.min())
        if first + 6 in true_new.index:
            picks.append(first + 6)
        q4 = [cm for cm in true_new.index if cm.month in (10, 11, 12)]
        if q4:
            picks.append(min(q4))
    if len(full):
        last_year = full[[cm.year == full.index.max().year for cm in full.index]]
        picks.append(last_year["CohortSize"].idxmax())
        picks.append(full.index.max())
    return sorted(set(picks))


def days_to_second_hist(customers: pd.DataFrame, bin_width: int = 7, cap: int = 365) -> pd.DataFrame:
    """Binned days-to-second-order for repeat customers.

    Day 0 (same-day second invoice) is its own bin so the spike stays visible; the rest are
    `bin_width`-day bins; everything at or beyond `cap` is pooled into a final 'cap+' bin.
    """
    d = customers.loc[customers["IsRepeat"], "DaysToSecond"].astype(int)
    rows = [{"bin_start": 0, "bin_end": 0, "label": "0", "customers": int((d == 0).sum())}]
    start = 1
    while start < cap:
        end = min(start + bin_width - 1, cap - 1)
        rows.append({"bin_start": start, "bin_end": end, "label": f"{start}-{end}",
                     "customers": int(d.between(start, end).sum())})
        start = end + 1
    rows.append({"bin_start": cap, "bin_end": int(max(d.max(), cap)), "label": f"{cap}+",
                 "customers": int((d >= cap).sum())})
    h = pd.DataFrame(rows)
    h["share"] = h["customers"] / len(d) if len(d) else 0.0
    return h


def cumulative_revenue_per_customer(orders: pd.DataFrame, bounds: dict) -> pd.DataFrame:
    """Cumulative revenue per cohort customer by cohort index (mean, so wholesale-skewed).
    Cells whose calendar month is not fully observed are NaN, as in `retention_matrix`."""
    last_complete = bounds["last_complete_month"]
    o = orders[orders["CohortMonth"] <= last_complete]
    rev = o.groupby(["CohortMonth", "CohortIndex"])["Revenue"].sum().unstack(1).sort_index()
    rev = rev.reindex(columns=sorted(rev.columns)).fillna(0.0)
    sizes = o.loc[o["OrderNumber"] == 1].groupby("CohortMonth").size()
    cum = rev.cumsum(axis=1).divide(sizes, axis=0)
    mask = pd.DataFrame([[cm + ci > last_complete for ci in cum.columns] for cm in cum.index],
                        index=cum.index, columns=cum.columns)
    return cum.mask(mask)


def segments_long(customers: pd.DataFrame, bounds: dict, window: int = 90,
                  by: tuple = SEGMENT_COLUMNS) -> pd.DataFrame:
    """All segment tables stacked into one long frame (one row per segment value)."""
    frames = []
    for col in by:
        t = segment_table(customers, bounds, col, window).reset_index()
        t = t.rename(columns={col: "segment_value", f"Repeat{window}d": f"repeat_{window}d",
                              "AvgOrders": "avg_orders", "MedianFirstOrderValue": "median_first_order_value",
                              "RevenuePerCustomer": "revenue_per_customer", "SmallSample": "small_sample"})
        t["segment_value"] = t["segment_value"].astype(str)
        t.insert(0, "segment_type", col)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def tier_bounds(customers: pd.DataFrame) -> pd.DataFrame:
    """Min / max first-order value (GBP) and customer count of each FirstOrderTier."""
    return (customers.groupby("FirstOrderTier", observed=True)["FirstOrderValue"]
            .agg(["min", "max", "size"]).rename(columns={"size": "customers"}))


def proportion_gap(n_a: int, p_a: float, n_b: int, p_b: float, z: float = 1.96) -> dict:
    """Difference between two independent proportions (a - b) in percentage points, with a
    normal-approximation 95% confidence interval. `distinguishable` is False when the interval
    spans zero, i.e. the gap is small relative to the sample sizes."""
    diff = p_a - p_b
    se = float(np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b))
    low, high = diff - z * se, diff + z * se
    return {"diff_pp": float(diff * 100), "ci_low_pp": float(low * 100), "ci_high_pp": float(high * 100),
            "distinguishable": bool(low > 0 or high < 0)}


#: Segment pairs compared in the write-up: (segment column, value a, value b) -> gap = a - b.
SEGMENT_PAIRS = (("Region", "UK", "International"), ("FirstOrderTier", "High", "Low"),
                 ("IsQ4Cohort", "True", "False"), ("CohortYear", "max", "min"))


def segment_gaps(segments: dict, window: int = 90) -> dict:
    """Gap and confidence interval for each pair in SEGMENT_PAIRS, from the nested segments dict
    produced by `headline_metrics`. 'max'/'min' pick the latest / earliest value of the segment."""
    key, out = f"repeat_{window}d", {}
    for seg, a, b in SEGMENT_PAIRS:
        vals = segments.get(seg, {})
        if a == "max" and vals:
            a, b = max(vals), min(vals)
        if a in vals and b in vals and a != b:
            out[seg] = {"a": a, "b": b, "n_a": vals[a]["n"], "n_b": vals[b]["n"],
                        **proportion_gap(vals[a]["n"], vals[a][key], vals[b]["n"], vals[b][key])}
    return out


def headline_metrics(orders: pd.DataFrame, customers: pd.DataFrame, bounds: dict,
                     window: int = 90) -> dict:
    """Every headline number, as plain Python types, in the `results.json` layout
    (sections: retention, repeat, revenue, segments)."""
    first_month, last_complete = bounds["first_month"], bounds["last_complete_month"]
    col = f"Repeat{window}d"

    R = retention_matrix(orders, bounds)
    avg = average_retention_table(R["pct"], first_month=first_month)
    drop = dropoff_table(avg["mean_retention"])
    worst = int(drop["Change"].idxmin())

    summ = cohort_summary(customers, bounds, window)
    full = summ.loc[summ["FullyObserved"], col]
    full_true_new = full[full.index > first_month]
    true_new_sizes = summ.loc[summ.index > first_month, "CohortSize"]
    q4_sizes = true_new_sizes[[cm.month in (10, 11, 12) for cm in true_new_sizes.index]]
    q4_full = [str(cm) for cm in full_true_new.index if cm.month in (10, 11, 12)]

    km = survival_to_second(customers)
    half = km.loc[km["repeated_by"] >= 0.5, "day"]
    repeaters = customers[customers["IsRepeat"]]
    rs = revenue_split(orders, bounds)
    repeat_rev = orders.loc[orders["OrderNumber"] > 1].groupby("CustomerID")["Revenue"].sum()
    second_orders = orders.loc[orders["OrderNumber"] == 2, "Revenue"]

    seg = segments_long(customers, bounds, window)
    segments = {
        t: {r["segment_value"]: {"n": int(r["n"]), f"repeat_{window}d": float(r[f"repeat_{window}d"]),
                                 "avg_orders": float(r["avg_orders"]),
                                 "median_first_order_value": float(r["median_first_order_value"]),
                                 "revenue_per_customer": float(r["revenue_per_customer"]),
                                 "small_sample": bool(r["small_sample"])}
            for _, r in g.iterrows()}
        for t, g in seg.groupby("segment_type", sort=False)}

    return {
        "retention": {
            "avg_curve": {str(int(k)): float(v) for k, v in avg["mean_retention"].items()},
            "avg_curve_n_cohorts": {str(int(k)): int(v) for k, v in avg["n_cohorts"].items()},
            "biggest_dropoff": {"month_index": worst,
                                "change_pp": float(drop.loc[worst, "Change"] * 100)},
            "n_cohorts": int(len(summ)),
            "n_fully_observed": int(summ["FullyObserved"].sum()),
        },
        "repeat": {
            "window_days": int(window),
            "repeat_window_mean": float(full.mean()),
            "repeat_window_min": float(full.min()),
            "repeat_window_max": float(full.max()),
            "repeat_window_mean_true_new": float(full_true_new.mean()),
            "repeat_window_min_true_new": float(full_true_new.min()),
            "repeat_window_max_true_new": float(full_true_new.max()),
            "n_fully_observed_true_new": int(len(full_true_new)),
            "median_days_to_second": float(repeaters["DaysToSecond"].median()),
            "same_day_share": float((repeaters["DaysToSecond"] == 0).mean()),
            "km_repeated_by": repeat_by_day(km, KM_DAYS),
            "km_day_half_repeated": int(half.iloc[0]) if len(half) else None,
            "weakest_true_new_cohort": str(full_true_new.idxmin()),
            "strongest_true_new_cohort": str(full_true_new.idxmax()),
            "q4_cohorts_fully_observed": q4_full,
            "n_customers": int(len(customers)),
            "n_repeat_customers": int(len(repeaters)),
            "repeat_ever_overall": float(customers["IsRepeat"].mean()),
            "repeat_ever_first_cohort": float(summ["RepeatEver"].iloc[0]),
            "repeat_ever_last_cohort": float(summ["RepeatEver"].iloc[-1]),
        },
        "revenue": {
            "repeat_share": float(rs["Repeat"].sum() / rs["Total"].sum()),
            "first_total": float(rs["First"].sum()),
            "repeat_total": float(rs["Repeat"].sum()),
            "total_complete_cohorts": float(rs["Total"].sum()),
            "first_cohort_total": float(rs["Total"].iloc[0]),
            "repeat_share_true_new": float(rs["Repeat"].iloc[1:].sum() / rs["Total"].iloc[1:].sum())
            if len(rs) > 1 else float("nan"),
            "median_repeat_revenue_per_repeat_customer": float(repeat_rev.median()),
            "mean_repeat_revenue_per_repeat_customer": float(repeat_rev.mean()),
            "avg_new_customers_per_month": float(true_new_sizes.mean()),
            "avg_new_customers_per_q4_month": float(q4_sizes.mean()) if len(q4_sizes) else float("nan"),
            "median_second_order_value": float(second_orders.median()),
            "median_first_order_value": float(customers["FirstOrderValue"].median()),
            "years_of_data": float((bounds["data_end"] - customers["FirstOrderDate"].min()).days / 365.25),
        },
        "segments": segments,
        "segment_gaps": segment_gaps(segments, window),
    }


def sensitivity_metrics(headline: dict, n_customers: int, revenue: float) -> dict:
    """The handful of headline numbers compared between cleaning variants (trap 4)."""
    return {
        "repeat_window_mean": headline["repeat"]["repeat_window_mean"],
        "km_repeated_by_window": headline["repeat"]["km_repeated_by"].get(
            str(headline["repeat"]["window_days"]), float("nan")),
        "median_days_to_second": headline["repeat"]["median_days_to_second"],
        "month1_retention": headline["retention"]["avg_curve"].get("1", float("nan")),
        "repeat_revenue_share": headline["revenue"]["repeat_share"],
        "customers": int(n_customers),
        "revenue": float(revenue),
    }


def build_recommendations(h: dict) -> list[dict]:
    """Recommendations with quantified, illustrative impact — every input comes from `h`
    (the output of `headline_metrics`) or from the labelled assumption constants above.

    Estimates use the MEDIAN repeat revenue per repeat customer (wholesale buyers skew the
    mean) and are gross revenue, not profit: there is no cost, margin or channel data.
    """
    rep, rev, ret, seg = h["repeat"], h["revenue"], h["retention"], h["segments"]
    w = rep["window_days"]
    key = f"repeat_{w}d"
    new_pm = rev["avg_new_customers_per_month"]
    med_rep = rev["median_repeat_revenue_per_repeat_customer"]
    km = rep["km_repeated_by"]
    recs = []

    # 1. Win-back timing --------------------------------------------------------------
    est = ILLUSTRATIVE_UPLIFT * new_pm * 12 * med_rep
    recs.append({
        "title": "Time a win-back message before the median return date",
        "finding": (f"Median time to a second order is {rep['median_days_to_second']:.0f} days; "
                    f"{km['30']:.1%} of customers have re-ordered by day 30 and {km['60']:.1%} by day 60 "
                    f"(Kaplan-Meier) - about {km['60'] / km['365']:.0%} of everyone who returns within a year."),
        "action": "Send a win-back email with a relevant re-order prompt at roughly day 35-45, "
                  "before the return curve flattens.",
        "formula": "uplift_pp x avg_new_customers_per_month x 12 x median_repeat_revenue",
        "inputs": {"uplift_pp": ILLUSTRATIVE_UPLIFT, "avg_new_customers_per_month": new_pm,
                   "months": 12, "median_repeat_revenue": med_rep},
        "estimate_gbp_per_year": round(est, 2),
        "assumptions": ("The 2 pp uplift is illustrative, not measured. Median repeat revenue is computed "
                        "over customers with 2+ orders and assumes converted customers behave like the "
                        "median existing repeater. Gross revenue, not profit."),
    })

    # 2. First-order value tier gap -----------------------------------------------------
    tiers = seg.get("FirstOrderTier", {})
    if {"Low", "Mid", "High"} <= set(tiers):
        n_all = sum(t["n"] for t in tiers.values())
        low_share = tiers["Low"]["n"] / n_all
        gap = tiers["Mid"][key] - tiers["Low"][key]
        est = GAP_CLOSED_SHARE * gap * low_share * new_pm * 12 * med_rep
        recs.append({
            "title": "Lift small first baskets towards the mid tier",
            "finding": (f"Repeat-within-{w}-days rises with first-order value: Low {tiers['Low'][key]:.1%} "
                        f"(n={tiers['Low']['n']:,}), Mid {tiers['Mid'][key]:.1%} (n={tiers['Mid']['n']:,}), "
                        f"High {tiers['High'][key]:.1%} (n={tiers['High']['n']:,}) - a "
                        f"{(tiers['High'][key] - tiers['Low'][key]) * 100:.1f} pp spread."),
            "action": "Give low-value first-time buyers a reason to build a bigger first basket "
                      "(free-delivery threshold, starter bundles), and prioritise acquisition channels "
                      "that bring mid/high first orders.",
            "formula": "gap_closed_share x (mid_rate - low_rate) x low_tier_share x "
                       "avg_new_customers_per_month x 12 x median_repeat_revenue",
            "inputs": {"gap_closed_share": GAP_CLOSED_SHARE, "mid_rate": tiers["Mid"][key],
                       "low_rate": tiers["Low"][key], "low_tier_share": low_share,
                       "avg_new_customers_per_month": new_pm, "months": 12,
                       "median_repeat_revenue": med_rep},
            "estimate_gbp_per_year": round(est, 2),
            "assumptions": ("Closing 25% of the Low-to-Mid gap is illustrative. The tier gap is a correlation: "
                            "large first orders partly identify wholesale buyers, so a bigger basket may not "
                            "cause a return visit. Gross revenue, not profit."),
        })

    # 3. Q4 cohort weakness -------------------------------------------------------------
    q4 = seg.get("IsQ4Cohort", {})
    if {"True", "False"} <= set(q4):
        gap = q4["False"][key] - q4["True"][key]
        q4_per_year = rev["avg_new_customers_per_q4_month"] * 3
        est = GAP_CLOSED_SHARE * gap * q4_per_year * med_rep
        recs.append({
            "title": "Give Q4-acquired customers their own follow-up",
            "finding": (f"Customers first acquired in Oct-Dec repeat within {w} days at {q4['True'][key]:.1%} "
                        f"(n={q4['True']['n']:,}) versus {q4['False'][key]:.1%} (n={q4['False']['n']:,}) for "
                        f"other cohorts - {gap * 100:.1f} pp lower."),
            "action": "Run a dedicated January-February re-activation sequence for Q4 first-time buyers "
                      "(non-seasonal ranges, spring catalogue) instead of treating them like any other cohort.",
            "formula": "gap_closed_share x (other_rate - q4_rate) x q4_new_customers_per_year x "
                       "median_repeat_revenue",
            "inputs": {"gap_closed_share": GAP_CLOSED_SHARE, "other_rate": q4["False"][key],
                       "q4_rate": q4["True"][key], "q4_new_customers_per_year": q4_per_year,
                       "median_repeat_revenue": med_rep},
            "estimate_gbp_per_year": round(est, 2),
            "assumptions": ("Closing 25% of the gap is illustrative. The Q4 rate rests on only "
                            f"{len(rep['q4_cohorts_fully_observed'])} fully observed Q4 cohorts "
                            f"({', '.join(rep['q4_cohorts_fully_observed'])}), i.e. a single season. Q4 new customers "
                            "per year = average true-new Q4 cohort size x 3 months. Seasonal gift buyers may "
                            "simply not need the product again within the window. Gross revenue, not profit."),
        })

    # 4. Month 0 -> 1 cliff --------------------------------------------------------------
    m1 = ret["avg_curve"].get("1")
    if m1 is not None:
        second = rev["median_second_order_value"]
        est = ILLUSTRATIVE_UPLIFT * new_pm * 12 * second
        recs.append({
            "title": "Attack the month-1 cliff with a second-order prompt",
            "finding": (f"The biggest drop-off is month {ret['biggest_dropoff']['month_index'] - 1} to "
                        f"{ret['biggest_dropoff']['month_index']}: average retention falls "
                        f"{abs(ret['biggest_dropoff']['change_pp']):.1f} pp, to {m1:.1%} in month 1, "
                        "and then stays roughly flat."),
            "action": "Put the retention budget into the first 30 days: an onboarding sequence and a "
                      "time-limited second-order incentive, rather than spreading effort across later months.",
            "formula": "uplift_pp x avg_new_customers_per_month x 12 x median_second_order_value",
            "inputs": {"uplift_pp": ILLUSTRATIVE_UPLIFT, "avg_new_customers_per_month": new_pm,
                       "months": 12, "median_second_order_value": second},
            "estimate_gbp_per_year": round(est, 2),
            "assumptions": ("The 2 pp uplift in month-1 retention is illustrative. Counts only the one extra "
                            "order (median second-order value); it targets the same customers as recommendation 1, "
                            "so the two overlap and must not be added. Gross revenue before any incentive cost."),
        })

    # 5. Repeat revenue concentration ----------------------------------------------------
    est = RETAINED_SHARE * rep["n_repeat_customers"] * med_rep / rev["years_of_data"]
    recs.append({
        "title": "Protect the repeat base that carries the revenue",
        "finding": (f"Repeat orders generate {rev['repeat_share']:.1%} of revenue; "
                    f"{rep['n_repeat_customers']:,} customers have ordered more than once."),
        "action": "Treat existing repeat customers as the core asset: lapse alerts when a regular buyer "
                  "goes quiet, account management for the largest, and service levels that prevent churn.",
        "formula": "retained_share x repeat_customers x median_repeat_revenue / years_of_data",
        "inputs": {"retained_share": RETAINED_SHARE, "repeat_customers": rep["n_repeat_customers"],
                   "median_repeat_revenue": med_rep, "years_of_data": rev["years_of_data"]},
        "estimate_gbp_per_year": round(est, 2),
        "assumptions": ("Value of keeping 1% of the repeat base from lapsing, at the median repeat revenue "
                        "annualised over the observation period - illustrative. Wholesale accounts are worth "
                        "far more than the median, so losing a large one costs much more. Gross revenue."),
    })
    return recs
