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
