import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")


app._unparsable_cell(
    r"""
    import warnings
    warnings.filterwarnings("ignore")
    import marimo as mo
    import duckdb as ddb
    import polars as pl
    import hvplot.polars
    import holoviews as hv
    import glob
    from operator import contains
    from pathlib import Path
    import gc
    import 
    """,
    name="_"
)


@app.cell
def _(glob, pl):
    lfs = [pl.scan_parquet(f, hive_partitioning=True) for f in glob.glob("backblaze_drive_stats/*2024/*.parquet", recursive=True)]
    df = pl.concat(lfs, how="diagonal_relaxed")
    df = df.with_columns(
        pl.col('date').str.to_date().alias('date'),
        pl.col('failure').cast(pl.Boolean).alias('failure'),
        pl.col('cluster_id').cast(pl.UInt8),
        pl.col('vault_id').cast(pl.UInt16),
        pl.col('pod_slot_num').cast(pl.Int16)
    )
    relevant_smart = [
        # Main failure indicators
        "smart_5_raw",
        "smart_187_raw",
        "smart_188_raw",
        "smart_197_raw",
        "smart_198_raw",
        "smart_9_raw",
        "smart_12_raw",
        "smart_193_raw",
        "smart_194_raw",
        "smart_199_raw",
    ]

    df_topk = pl.read_csv('./backblaze_drive_stats/top_20.csv')
    df = df.filter(
        pl.col('model').is_in(df_topk.top_k(3, by='count')['model'])
    )
    return (df,)


@app.cell(hide_code=True)
def _(df, pl):
    GAP_TOLERANCE = 3   # absences of <= 3 days are ignored / bridged

    base = (
        df.select(["date", "serial_number", "model", "failure"])
          .with_columns(pl.col("failure").cast(pl.Boolean))
          .unique(subset=["serial_number", "date"], keep="first")   # kill duplicate snapshots
          .sort(["serial_number", "date"])
    )

    STUDY_START = base.select(pl.col("date").min()).collect().item()
    STUDY_END   = base.select(pl.col("date").max()).collect().item()

    episodes = (
        base
        .with_columns(
            previous_date=pl.col("date").shift(1).over("serial_number", order_by="date"),
            previous_failure=pl.col("failure")
                .shift(1).over("serial_number", order_by="date")
                .fill_null(False),
        )
        .with_columns(
            gap_days=(pl.col("date") - pl.col("previous_date")).dt.total_days()
        )
        .with_columns(
            new_episode=(
                pl.col("previous_date").is_null()
                | pl.col("previous_failure")
                | (pl.col("gap_days") > GAP_TOLERANCE)      # <- tolerance
            )
        )
        .with_columns(
            episode_id=pl.col("new_episode").cast(pl.UInt32)
                .cum_sum().over("serial_number", order_by="date") - 1
        )
    )

    spells = (
        episodes
        .group_by(["serial_number", "episode_id"], maintain_order=True)
        .agg(
            model=pl.first("model"),
            first_date=pl.min("date"),
            last_date=pl.max("date"),
            event=pl.col("failure").any(),
            observed_days=pl.len(),
        )
        .sort(["serial_number", "episode_id"])
        .with_columns(
            duration_days=(pl.col("last_date") - pl.col("first_date")).dt.total_days() + 1,
            # gap that separates this episode from the previous one (null for episode 0)
            gap_before_days=(
                pl.col("first_date")
                - pl.col("last_date").shift(1).over("serial_number", order_by="episode_id")
            ).dt.total_days(),
            prev_event=pl.col("event")
                .shift(1).over("serial_number", order_by="episode_id")
                .fill_null(False),
            n_episodes=pl.len().over("serial_number"),
        )
        .with_columns(
            missing_days=pl.col("duration_days") - pl.col("observed_days"),

            # (2) this episode is a reappearance AFTER a recorded failure
            reappeared_after_failure=pl.col("prev_event"),

            # reappearance after a long absence, without a failure in between
            reappeared_after_gap=(
                pl.col("gap_before_days").is_not_null() & ~pl.col("prev_event")
            ),

            # (3) this episode ends without a failure -> right-censored
            censored=~pl.col("event"),
        )
        .with_columns(
            censoring_reason=pl.when(pl.col("event")).then(None)
                .when(pl.col("last_date") >= STUDY_END).then(pl.lit("administrative"))
                .otherwise(pl.lit("dropout"))
        )
    )

    unit_flags = spells.group_by("serial_number").agg(
        n_episodes=pl.len(),
        n_failures=pl.col("event").sum(),
        any_resurrection=pl.col("reappeared_after_failure").any(),
    )

    units = (
        spells.filter(pl.col("episode_id") == 0)
        .join(unit_flags, on="serial_number", how="left")
        .with_columns(
            duration=pl.col("duration_days"),
            status=(
                pl.when(pl.col("any_resurrection"))
                  .then(pl.lit(None, dtype=pl.Utf8))            # "nothing" -> unusable
                .when(pl.col("event"))
                  .then(pl.lit("died"))
                .otherwise(pl.lit("censored"))
            ),
        )
        .with_columns(
            why=(
                pl.when(pl.col("status").is_null()).then(pl.lit("post_failure_reappearance"))
                .when(pl.col("status") == "died").then(pl.lit("failure"))
                .when(pl.col("n_episodes") > 1).then(pl.lit("long_absence"))       # gap > tolerance
                .when(pl.col("last_date") >= STUDY_END).then(pl.lit("administrative"))
                .otherwise(pl.lit("dropout"))
            ),
            left_truncated=pl.col("first_date") <= STUDY_START,
        )
        .select([
            "serial_number", "model", "first_date", "last_date",
            "duration", "observed_days", "missing_days",
            "status", "why", "left_truncated",
            "n_episodes", "n_failures", "any_resurrection",
        ])
    )

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
