"""tests/test_core.py — behaviour of the cohort pipeline on tiny, hand-checkable data."""
import numpy as np
import pandas as pd
import pytest

from retention import core as m


def rows(*recs):
    """Build a raw-shaped frame from (invoice, code, qty, date, price, customer, country)."""
    return pd.DataFrame(
        [dict(zip(["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID", "Country"], r))
         for r in recs]
    ).astype({"Invoice": str, "StockCode": str}).assign(InvoiceDate=lambda d: pd.to_datetime(d.InvoiceDate))


@pytest.fixture
def toy():
    return rows(
        # c1: Jan order, two invoices in Feb (same month -> 2 orders, 1 active customer)
        ("1001", "22041", 2, "2010-01-05", 10.0, 1, "United Kingdom"),
        ("1002", "22041", 1, "2010-02-10", 10.0, 1, "United Kingdom"),
        ("1003", "22041", 1, "2010-02-20", 10.0, 1, "United Kingdom"),
        # c2: Jan only, never repeats
        ("1004", "22041", 1, "2010-01-20", 50.0, 2, "France"),
        # c3: Feb cohort, repeats in April (skips March)
        ("1005", "79323P", 1, "2010-02-02", 20.0, 3, "United Kingdom"),
        ("1006", "79323P", 1, "2010-04-02", 20.0, 3, "United Kingdom"),
        # rows that must be dropped
        ("C999", "22041", 1, "2010-01-06", 10.0, 1, "United Kingdom"),   # cancellation
        ("1007", "POST", 1, "2010-01-07", 18.0, 1, "United Kingdom"),    # non-product
        ("1008", "22041", -3, "2010-01-08", 10.0, 2, "France"),          # bad quantity
        ("1009", "22041", 1, "2010-01-09", 0.0, 2, "France"),            # zero price
        ("1010", "22041", 1, "2010-01-10", 10.0, np.nan, "United Kingdom"),  # no customer
    )


@pytest.fixture
def built(toy):
    lines, log = m.clean(toy)
    b = m.period_bounds(lines)
    o = m.build_orders(lines)
    c = m.build_customers(o, b)
    return lines, log, b, o, c


def test_cleaning_drops_exactly_the_bad_rows(built):
    lines, log, *_ = built
    assert len(lines) == 6
    assert log["raw_rows"] == 11 and log["customers"] == 3
    assert not lines.Invoice.str.upper().str.startswith(("C", "A")).any()
    assert (lines.Quantity > 0).all() and (lines.Price > 0).all()
    assert lines.CustomerID.notna().all()
    assert "POST" in log["non_product_codes"]


def test_revenue_is_quantity_times_price(built):
    lines, *_ = built
    assert lines.loc[lines.Invoice == "1001", "Revenue"].iloc[0] == 20.0


def test_orders_are_one_row_per_invoice(built):
    *_, o, c = built
    assert len(o) == 6
    assert c.loc[1, "Orders"] == 3          # 1001, 1002, 1003
    assert c.loc[2, "Orders"] == 1


def test_cohort_month_and_index(built):
    *_, o, c = built
    assert c.loc[1, "CohortMonth"] == pd.Period("2010-01", "M")
    assert c.loc[3, "CohortMonth"] == pd.Period("2010-02", "M")
    idx = o.set_index("Invoice")["CohortIndex"]
    assert idx["1001"] == 0 and idx["1002"] == 1 and idx["1003"] == 1
    assert idx["1006"] == 2                 # Feb -> Apr skips a month


def test_two_invoices_same_month_count_once_in_retention(built):
    *_, b, o, _ = built
    R = m.retention_matrix(o, b)
    jan = pd.Period("2010-01", "M")
    assert R["sizes"][jan] == 2             # customers 1 and 2
    assert R["counts"].loc[jan, 1] == 1     # customer 1 active once, despite 2 invoices
    assert R["pct"].loc[jan, 1] == 0.5


def test_month_zero_is_always_full(built):
    *_, b, o, _ = built
    R = m.retention_matrix(o, b)
    assert (R["pct"][0].dropna() == 1).all()


def test_repeat_flags_and_days_to_second(built):
    *_, c = built
    assert bool(c.loc[1, "IsRepeat"]) and not bool(c.loc[2, "IsRepeat"])
    assert c.loc[1, "DaysToSecond"] == 36            # 05 Jan -> 10 Feb
    assert pd.isna(c.loc[2, "DaysToSecond"])
    assert c.loc[3, "DaysToSecond"] == 59            # 02 Feb -> 02 Apr


def test_first_cohort_is_marked_left_censored(built):
    *_, c = built
    assert not c.loc[1, "IsTrueNew"] and not c.loc[2, "IsTrueNew"]   # Jan is the first month
    assert c.loc[3, "IsTrueNew"]


def test_window_metric_only_fills_fully_observed_cohorts(built):
    *_, b, o, c = built
    s = m.cohort_summary(c, b, window=90)
    # data ends 2010-04-02, so no cohort has a full 90-day window after its last day
    assert not s["FullyObserved"].any()
    assert s["Repeat90d"].isna().all()
    s7 = m.cohort_summary(c, b, window=7)
    assert s7.loc[pd.Period("2010-01", "M"), "FullyObserved"]


def test_retention_cells_beyond_data_are_masked(built):
    *_, b, o, _ = built
    R = m.retention_matrix(o, b)
    feb = pd.Period("2010-02", "M")
    # data ends 2010-04-02 so the last complete month is March; Feb+2 = April must be NaN
    assert b["last_complete_month"] == pd.Period("2010-03", "M")
    assert pd.isna(R["pct"].loc[feb, 2])


def test_last_complete_month_when_data_ends_on_month_end():
    df = rows(("1", "22041", 1, "2010-01-05", 10.0, 1, "UK"),
              ("2", "22041", 1, "2010-03-31", 10.0, 1, "UK"))
    lines, _ = m.clean(df)
    assert m.period_bounds(lines)["last_complete_month"] == pd.Period("2010-03", "M")


def test_survival_curve_is_monotonic_and_bounded(built):
    *_, c = built
    km = m.survival_to_second(c)
    assert km["repeated_by"].is_monotonic_increasing
    assert km["repeated_by"].between(0, 1).all()


def test_revenue_split_totals_match_orders(built):
    *_, b, o, _ = built
    rs = m.revenue_split(o, b)
    expected = o[o.CohortMonth <= b["last_complete_month"]].Revenue.sum()
    assert rs["Total"].sum() == pytest.approx(expected)


def test_segment_table_reports_sample_size(built):
    *_, b, _, c = built
    t = m.segment_table(c, b, "Region", window=7)
    assert "n" in t.columns and t["SmallSample"].all()
