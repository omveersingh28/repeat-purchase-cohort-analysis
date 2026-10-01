"""One command rebuilds everything:  python -m src.run_pipeline [--data-path PATH] [--window DAYS]

Outputs
-------
data/processed/lines.parquet   cleaned line items (gitignored)
reports/tables/*.csv           every table quoted in the README / notebooks
reports/figures/*.png          the ten report figures
reports/results.json           every headline number (single source of truth)
app_data/*                     small aggregates the dashboard reads (no raw rows)

The run is idempotent: a second run changes nothing but the `generated_at` timestamp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.retention import config, core, plots

#: A change in the headline repeat rate above this (in percentage points) counts as material.
SENSITIVITY_THRESHOLD_PP = 1.0


@contextmanager
def step(name: str):
    """Log a pipeline step with its wall-clock duration."""
    start = time.perf_counter()
    print(f"[ .. ] {name}", flush=True)
    yield
    print(f"[done] {name}  ({time.perf_counter() - start:.1f}s)", flush=True)


def sha256_first_mb(path: Path) -> str:
    """SHA-256 of the first 1 MB of a file — a cheap fingerprint of which dataset was used."""
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read(1024 * 1024)).hexdigest()


def jsonable(obj):
    """Recursively convert to strict-JSON types (NaN -> null, numpy scalars -> Python)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        obj = obj.item()
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def cohort_csv(df: pd.DataFrame | pd.Series, path: Path, **kw) -> None:
    """Write a cohort-indexed frame with the cohort month as a plain 'YYYY-MM' string."""
    out = df.copy()
    out.index = out.index.astype(str)
    out.index.name = "CohortMonth"
    out.to_csv(path, **kw)


def pct_matrix(pct: pd.DataFrame) -> pd.DataFrame:
    """Retention fractions -> percentages, rounded for a compact CSV."""
    return (pct * 100).round(4)


def run(data_path: Path | None = None, window: int = 90) -> dict:
    """Run the full pipeline and return the results dictionary written to results.json."""
    t0 = time.perf_counter()
    config.ensure_dirs()
    T, F, A = config.TABLES, config.FIGURES, config.APP_DATA
    wcol = f"Repeat{window}d"

    # ------------------------------------------------------------------ load & clean
    dataset = Path(data_path).resolve() if data_path else config.find_dataset()
    with step(f"Load raw data from {config.relative_to_root(dataset)}"):
        raw = core.load_raw(dataset)
    with step("Clean (4 rules) and save data/processed/lines.parquet"):
        lines, log = core.clean(raw)
        lines.to_parquet(config.LINES_PARQUET, index=False)
        bounds = core.period_bounds(lines)
        missing_customer_rows = int(raw["CustomerID"].isna().sum())

    # ------------------------------------------------------------------ build
    with step("Build orders, customers, cohorts"):
        orders = core.build_orders(lines)
        customers = core.build_customers(orders, bounds)
        R = core.retention_matrix(orders, bounds)
        R_uk = core.retention_matrix(orders, bounds, customers.index[customers["Region"] == "UK"])
        R_intl = core.retention_matrix(orders, bounds, customers.index[customers["Region"] == "International"])
        avg = core.average_retention_table(R["pct"], first_month=bounds["first_month"])
        dropoff = core.dropoff_table(avg["mean_retention"])
        summary = core.cohort_summary(customers, bounds, window)
        km = core.survival_to_second(customers)
        hist = core.days_to_second_hist(customers)
        split = core.revenue_split(orders, bounds)
        cum_rev = core.cumulative_revenue_per_customer(orders, bounds)
        segments = core.segments_long(customers, bounds, window)
        tiers = core.tier_bounds(customers)
        waterfall = core.waterfall_table(log)
        headline = core.headline_metrics(orders, customers, bounds, window)
        recommendations = core.build_recommendations(headline)

    # ------------------------------------------------------------------ sensitivity (trap 4)
    with step("Sensitivity check: keep non-product stock codes"):
        lines_all, log_all = core.clean(raw, drop_non_product=False)
        bounds_all = core.period_bounds(lines_all)
        orders_all = core.build_orders(lines_all)
        customers_all = core.build_customers(orders_all, bounds_all)
        headline_all = core.headline_metrics(orders_all, customers_all, bounds_all, window)
        non_product = core.non_product_breakdown(lines_all)
        sens_true = core.sensitivity_metrics(headline, log["customers"], log["revenue"])
        sens_false = core.sensitivity_metrics(headline_all, log_all["customers"], log_all["revenue"])
        diff_pp = (sens_false["repeat_window_mean"] - sens_true["repeat_window_mean"]) * 100
    del raw, lines_all

    # ------------------------------------------------------------------ results.json
    with step("Write reports/results.json"):
        results = {
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": {"path": config.relative_to_root(dataset),
                        "sha256_first_1mb": sha256_first_mb(dataset),
                        "raw_rows": log["raw_rows"]},
            "cleaning": {**{k: log[k] for k in (
                "raw_rows", "after_missing_customer", "after_cancel_adjust", "after_qty_price",
                "after_non_product", "non_product_rows", "non_product_codes",
                "non_product_revenue_share", "customers", "invoices", "revenue", "date_min", "date_max")},
                "missing_customer_rows": missing_customer_rows,
                "missing_customer_share": missing_customer_rows / log["raw_rows"],
                "customers_lost_to_non_product": log_all["customers"] - log["customers"],
                "rows_kept_share": log["after_non_product"] / log["raw_rows"],
                "waterfall": [{"step": r["Step"], "rows_remaining": int(r["RowsRemaining"]),
                               "rows_removed": int(r["RowsRemoved"]),
                               "pct_of_raw_removed": float(r["PctOfRawRemoved"])}
                              for _, r in waterfall.iterrows()]},
            "bounds": {"first_month": str(bounds["first_month"]),
                       "last_complete_month": str(bounds["last_complete_month"]),
                       "data_end": str(bounds["data_end"])},
            "retention": headline["retention"],
            "repeat": headline["repeat"],
            "revenue": headline["revenue"],
            "segments": headline["segments"],
            "segment_gaps": headline["segment_gaps"],
            "first_order_tier_bounds": {k: {"min": r["min"], "max": r["max"], "customers": int(r["customers"])}
                                        for k, r in tiers.iterrows()},
            "sensitivity": {"drop_non_product_true": sens_true, "drop_non_product_false": sens_false,
                            "repeat_window_mean_diff_pp": diff_pp,
                            "threshold_pp": SENSITIVITY_THRESHOLD_PP,
                            "material": bool(abs(diff_pp) > SENSITIVITY_THRESHOLD_PP)},
            "recommendations": recommendations,
        }
        results = jsonable(results)
        config.RESULTS_JSON.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")

    # ------------------------------------------------------------------ tables
    with step("Write reports/tables/*.csv"):
        waterfall.to_csv(T / "cleaning_waterfall.csv", index=False)
        non_product.to_csv(T / "non_product_codes.csv")
        core.top_countries(customers).to_csv(T / "top_countries.csv")
        cohort_csv(core.monthly_orders(orders).rename_axis("OrderMonth"), T / "monthly_orders.csv")
        cohort_csv(pct_matrix(R["pct"]), T / "retention_pct.csv")
        cohort_csv(R["counts"].astype("Int64"), T / "retention_counts.csv")
        avg.to_csv(T / "avg_retention_curve.csv")
        dropoff.rename_axis("month_index").to_csv(T / "dropoff_table.csv")
        cohort_csv(summary, T / "cohort_summary.csv")
        km.to_csv(T / "survival_to_second.csv", index=False)
        pd.DataFrame({"day": [int(d) for d in headline["repeat"]["km_repeated_by"]],
                      "repeated_by": list(headline["repeat"]["km_repeated_by"].values())}
                     ).to_csv(T / "km_readings.csv", index=False)
        hist.to_csv(T / "days_to_second_hist.csv", index=False)
        cohort_csv(split, T / "revenue_split.csv")
        cohort_csv(cum_rev.round(2), T / "cumulative_revenue_per_customer.csv")
        segments.to_csv(T / "segments.csv", index=False)
        tiers.to_csv(T / "first_order_tier_bounds.csv")
        pd.DataFrame({"drop_non_product_true": sens_true, "drop_non_product_false": sens_false}
                     ).rename_axis("metric").to_csv(T / "sensitivity.csv")
        pd.DataFrame([{k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in r.items()}
                      for r in recommendations]).to_csv(T / "recommendations.csv", index=False)

    # ------------------------------------------------------------------ figures
    with step("Write reports/figures/*.png"):
        rep = headline["repeat"]
        figures = {
            "01_cleaning_waterfall": plots.cleaning_waterfall(waterfall),
            "02_retention_heatmap": plots.retention_heatmap(R["pct"], R["sizes"], bounds["first_month"]),
            "03_retention_curves": plots.retention_curves(
                R["pct"], core.select_cohorts(summary, bounds, window), bounds["first_month"]),
            "04_avg_retention_curve": plots.avg_retention_curve(avg),
            f"05_repeat_{window}d_by_cohort": plots.repeat_window_by_cohort(summary, bounds["first_month"], window),
            "06_survival_to_second": plots.survival_to_second(km, rep["km_repeated_by"]),
            "07_days_to_second_hist": plots.days_to_second_hist(
                hist, rep["median_days_to_second"], rep["same_day_share"]),
            "08_revenue_split": plots.revenue_split(split, bounds["first_month"]),
            "09_segment_repeat_rates": plots.segment_repeat_rates(segments, window),
            "10_first_order_tier": plots.first_order_tier(segments, tiers, window),
        }
        for name, fig in figures.items():
            plots.save(fig, F / f"{name}.png")

    # ------------------------------------------------------------------ dashboard data
    with step("Write app_data/ (dashboard aggregates)"):
        cohort_csv(pct_matrix(R["pct"]), A / "retention_pct.csv")
        cohort_csv(R["counts"].astype("Int64"), A / "retention_counts.csv")
        cohort_csv(pct_matrix(R_uk["pct"]), A / "retention_pct_uk.csv")
        cohort_csv(pct_matrix(R_intl["pct"]), A / "retention_pct_intl.csv")
        cohort_csv(R_uk["counts"].astype("Int64"), A / "retention_counts_uk.csv")
        cohort_csv(R_intl["counts"].astype("Int64"), A / "retention_counts_intl.csv")
        cohort_csv(summary, A / "cohort_summary.csv")
        sizes = pd.DataFrame({"CohortSize": R["sizes"].astype(int),
                              "IsTrueNew": R["sizes"].index > bounds["first_month"]})
        cohort_csv(sizes, A / "cohort_sizes.csv")
        avg.to_csv(A / "avg_retention_curve.csv")
        km[["day", "at_risk", "events", "repeated_by"]].to_csv(A / "survival_to_second.csv", index=False)
        hist.to_csv(A / "days_to_second_hist.csv", index=False)
        cohort_csv(split, A / "revenue_split.csv")
        segments[["segment_type", "segment_value", "n", f"repeat_{window}d", "avg_orders",
                  "median_first_order_value", "small_sample"]].to_csv(A / "segments.csv", index=False)
        shutil.copyfile(config.RESULTS_JSON, A / "results.json")

    # ------------------------------------------------------------------ summary
    c, rt, rp, rv = results["cleaning"], results["retention"], results["repeat"], results["revenue"]
    print("\n=== Headline numbers ===")
    print(f"rows {c['raw_rows']:,} -> {c['after_non_product']:,} | orders {c['invoices']:,} | "
          f"customers {c['customers']:,} | revenue £{c['revenue']:,.0f}")
    print(f"dates {c['date_min']} -> {c['date_max']} | last complete month {results['bounds']['last_complete_month']} | "
          f"cohorts {rt['n_cohorts']} / fully observed {rt['n_fully_observed']}")
    print(f"month-1 retention {rt['avg_curve']['1']:.1%} | month-2 {rt['avg_curve']['2']:.1%} | biggest drop-off "
          f"month {rt['biggest_dropoff']['month_index']} ({rt['biggest_dropoff']['change_pp']:+.1f} pp)")
    print(f"repeat-within-{window}d mean {rp['repeat_window_mean']:.1%} "
          f"(range {rp['repeat_window_min']:.1%}-{rp['repeat_window_max']:.1%}) | "
          f"KM {', '.join(f'd{k} {v:.1%}' for k, v in rp['km_repeated_by'].items() if v is not None)}")
    print(f"median days to 2nd order {rp['median_days_to_second']:.0f} | same-day share {rp['same_day_share']:.1%} | "
          f"repeat revenue share {rv['repeat_share']:.1%} | new customers/month {rv['avg_new_customers_per_month']:.0f}")
    print(f"sensitivity: keeping non-product codes moves repeat-{window}d by {diff_pp:+.2f} pp "
          f"({'MATERIAL' if results['sensitivity']['material'] else 'immaterial'})")
    print(f"\nPipeline finished in {time.perf_counter() - t0:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild every table, figure and dashboard file.")
    parser.add_argument("--data-path", type=Path, default=None,
                        help="Path to the raw dataset (default: auto-discover under the project root).")
    parser.add_argument("--window", type=int, default=90,
                        help="Repeat-purchase window in days for the headline metric (default: 90).")
    args = parser.parse_args()
    run(args.data_path, args.window)


if __name__ == "__main__":
    main()
