import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")


@app.cell
def _():
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

    return Path, gc, glob, hv, pl


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
    df.columns
    return df, relevant_smart


@app.cell
def _(df):
    df.select([
        "date",
    "serial_number",
    "model",
    "failure",
    ]).collect()
    return


@app.cell(hide_code=True)
def _():
    #episodes = (
    #    df.select(['date', 'serial_number', 'model', 'failure'])
    #    .sort(['serial_number', 'date'])
    #    .with_columns(
    #        prev_failure=pl.col('failure').shift(1).over('serial_number').fill_null(False)
    #    )
    #    .with_columns(
    #        episode_id=pl.col('prev_failure').cast(pl.Int32).cum_sum().over('serial_number')
    #    )
    #)
    #
    ## one row per episode (spell between failures) = your survival table
    #survival_df = (
    #    episodes
    #    .group_by(['serial_number', 'episode_id'], maintain_order=True)
    #    .agg(
    #        model=pl.first('model'),
    #        first_date=pl.first('date'),
    #        last_date=pl.last('date'),
    #        event=pl.col('failure').any(),   # True = failed, False = censored
    #    )
    #    .with_columns(
    #        duration_days=(pl.col('last_date') - pl.col('first_date')).dt.total_days() + 1
    #    )
    #)
    return


@app.cell(hide_code=True)
def _(GAP_TOLERANCE, MAX_BRIDGED, MAX_SPELLS, df, pl):
    daily = (
        df
        .with_columns(
            pl.col("failure").cast(pl.Boolean),
            pl.col("smart_9_raw").cast(pl.Int64, strict=False),
        )
        .unique(
            subset=["serial_number", "date"],
            keep="first",
        )
        .sort(["serial_number", "date"])
        .with_columns(
            gap_days=(
                pl.col("date")
                - pl.col("date")
                .shift(1)
                .over("serial_number", order_by="date")
            ).dt.total_days(),

            prev_failure=(
                pl.col("failure")
                .shift(1)
                .over("serial_number", order_by="date")
                .fill_null(False)
            ),
        )
        .with_columns(
            bridged_gap=pl.col("gap_days").is_between(
                2,
                GAP_TOLERANCE,
                closed="both",
            ),

            breaks_spell=(
                pl.col("gap_days").is_null()
                | pl.col("prev_failure")
                | (pl.col("gap_days") > GAP_TOLERANCE)
            ),
        )
        .with_columns(
            spell_id=(
                pl.col("breaks_spell")
                .cast(pl.UInt32)
                .cum_sum()
                .over("serial_number", order_by="date")
                - 1
            ),
        )
    )


    # -------------------------------------------------------------------
    # 2. Observation spells
    # -------------------------------------------------------------------

    spells = (
        daily
        .group_by(["serial_number", "spell_id"])
        .agg(
            model=pl.first("model"),
            first_date=pl.min("date"),
            last_date=pl.max("date"),
            event=pl.col("failure").any(),
            observed_days=pl.len(),
            n_bridged_gaps=pl.col("bridged_gap").sum(),
        )
        .sort(["serial_number", "spell_id"])
        .with_columns(
            duration_days=(
                pl.col("last_date") - pl.col("first_date")
            ).dt.total_days() + 1,

            after_failure=(
                pl.col("event")
                .shift(1)
                .over("serial_number", order_by="spell_id")
                .fill_null(False)
            ),
        )
        .with_columns(
            missing_days=(
                pl.col("duration_days") - pl.col("observed_days")
            ),
        )
    )


    # -------------------------------------------------------------------
    # 3. Study boundaries
    # -------------------------------------------------------------------

    STUDY_START = (
        daily
        .select(pl.col("date").min())
        .collect()
        .item()
    )

    STUDY_END = (
        daily
        .select(pl.col("date").max())
        .collect()
        .item()
    )


    # -------------------------------------------------------------------
    # 4. Historical diagnostics
    # -------------------------------------------------------------------

    summary = (
        spells
        .group_by("serial_number")
        .agg(
            n_spells=pl.len(),
            n_bridged_gaps=pl.col("n_bridged_gaps").sum(),
            n_failures=pl.col("event").sum(),
            resurrected=pl.col("after_failure").any(),
            total_missing_days=pl.col("missing_days").sum(),
        )
    )


    # -------------------------------------------------------------------
    # 5. Survival endpoint per drive
    # -------------------------------------------------------------------

    unit_endpoints = (
        daily
        .group_by("serial_number")
        .agg(
            model=pl.col("model").sort_by("date").first(),
            first_date=pl.col("date").min(),

            failure_date=(
                pl.col("date")
                .filter(pl.col("failure"))
                .min()
            ),

            final_observed_date=pl.col("date").max(),
            n_observed_total=pl.len(),
        )
        .with_columns(
            event=pl.col("failure_date").is_not_null(),

            # Failure takes priority over censoring.
            last_date=pl.coalesce(
                ["failure_date", "final_observed_date"]
            ),
        )
        .with_columns(
            duration=(
                pl.col("last_date") - pl.col("first_date")
            ).dt.total_days() + 1,

            status=(
                pl.when(pl.col("event"))
                .then(pl.lit("died"))
                .otherwise(pl.lit("censored"))
            ),

            censoring_reason=(
                pl.when(pl.col("event"))
                .then(pl.lit(None, dtype=pl.String))
                .when(pl.col("final_observed_date") >= STUDY_END)
                .then(pl.lit("administrative"))
                .otherwise(pl.lit("dropout"))
            ),
        )
    )


    # -------------------------------------------------------------------
    # 6. SMART 9 endpoints
    # -------------------------------------------------------------------

    smart_endpoints = (
        daily
        .join(
            unit_endpoints.select(
                "serial_number",
                "last_date",
            ),
            on="serial_number",
            how="inner",
        )
        # Ignore observations after the first failure.
        .filter(pl.col("date") <= pl.col("last_date"))
        .group_by("serial_number")
        .agg(
            observed_days=pl.len(),

            smart9_observed_rows=(
                pl.col("smart_9_raw")
                .is_not_null()
                .sum()
            ),

            entry_hours=(
                pl.col("smart_9_raw")
                .filter(pl.col("smart_9_raw").is_not_null())
                .sort_by("date")
                .first()
            ),

            exit_hours=(
                pl.col("smart_9_raw")
                .filter(pl.col("smart_9_raw").is_not_null())
                .sort_by("date")
                .last()
            ),

            smart9_first_date=(
                pl.col("date")
                .filter(pl.col("smart_9_raw").is_not_null())
                .min()
            ),

            smart9_last_date=(
                pl.col("date")
                .filter(pl.col("smart_9_raw").is_not_null())
                .max()
            ),
        )
        .with_columns(
            smart_duration_hours=(
                pl.col("exit_hours") - pl.col("entry_hours")
            ),

            smart9_available=(
                pl.col("entry_hours").is_not_null()
                & pl.col("exit_hours").is_not_null()
            ),
        )
    )


    # -------------------------------------------------------------------
    # 7. Final one-row-per-drive dataset
    # -------------------------------------------------------------------

    units = (
        unit_endpoints
        .join(
            summary,
            on="serial_number",
            how="left",
        )
        .join(
            smart_endpoints,
            on="serial_number",
            how="left",
        )
        .with_columns(
            left_truncated=pl.col("first_date") == STUDY_START,

            flaky=pl.col("n_bridged_gaps") > MAX_BRIDGED,
            interrupted=pl.col("n_spells") > MAX_SPELLS,
        )
        .with_columns(
            clean=~(
                pl.col("flaky")
                | pl.col("resurrected")
            ),
        )
        .select(
            "serial_number",
            "model",

            "first_date",
            "last_date",
            "failure_date",
            "final_observed_date",

            "duration",
            "observed_days",
            "status",
            "event",
            "censoring_reason",
            "left_truncated",

            "entry_hours",
            "exit_hours",
            "smart_duration_hours",
            "smart9_first_date",
            "smart9_last_date",
            "smart9_observed_rows",
            "smart9_available",

            "n_spells",
            "n_bridged_gaps",
            "n_failures",
            "total_missing_days",

            "flaky",
            "interrupted",
            "resurrected",
            "clean",
        )
    )
    return STUDY_END, spells, units


@app.cell
def _(episodes):
    episodes.columns
    return


@app.cell
def _(STUDY_END, pl, spells):
    unit_flags = spells.group_by("serial_number").agg(
        n_episodes=pl.len(),
        n_failures=pl.col("event").sum(),
        any_resurrection=pl.col("reappeared_after_failure").any(),
    )
    units = (
        spells.filter(pl.col("episode_id") == 0)
        .join(unit_flags, on="serial_number", how="left")
        .with_columns(
            duration=pl.col("duration_poh") / 24.0,     # <- power-on days, not calendar days
            entry_poh=pl.col("poh_start"),               # age (hours) already accrued when first seen
            exit_poh=pl.col("poh_end"),
            status=(
                pl.when(pl.col("any_resurrection"))
                  .then(pl.lit(None, dtype=pl.Utf8))
                .when(pl.col("event"))
                  .then(pl.lit("died"))
                .otherwise(pl.lit("censored"))
            ),
        )
        .with_columns(
            why=(
                pl.when(pl.col("status").is_null()).then(pl.lit("post_failure_reappearance"))
                .when(pl.col("status") == "died").then(pl.lit("failure"))
                .when(pl.col("n_episodes") > 1).then(pl.lit("long_absence"))
                .when(pl.col("last_date") >= STUDY_END).then(pl.lit("administrative"))
                .otherwise(pl.lit("dropout"))
            ),
            left_truncated=pl.col("entry_poh") > 0,   # drive already had hours on it when first seen
        )
        .select([
            "serial_number", "model", "first_date", "last_date",
            "duration", "duration_days", "observed_days", "missing_days",
            "entry_poh", "exit_poh",
            "status", "why", "left_truncated",
            "n_episodes", "n_failures", "any_resurrection",
        ])
    )
    return (units,)


@app.cell
def _(units):
    units.columns()
    return


@app.cell
def _(spells):
    spells.collect()
    return


@app.cell
def _(df, pl):
    df.select(["date", "serial_number", "model", "capacity_bytes", "failure", "datacenter", "cluster_id", "vault_id", "pod_id",
   
    "pod_slot_num"]).filter(
        pl.col('serial_number') == "ZL2P74BA"
    ).collect()
    return


@app.cell(hide_code=True)
def _(hv, pl, spells, survival_df):
    import pandas as pd

    hv.extension("bokeh")
    ids_ = spells.filter(
        ( pl.col('n_episodes') == 4 ).any().over('serial_number')
    ).collect()['serial_number']

    sdf = survival_df.filter(
        pl.col('serial_number').is_in(ids_)
    ).collect()

    # Sample drives, then retain all episodes belonging to those drives.
    n_drives = min(100, sdf.get_column("serial_number").n_unique())

    sample_serials = (
        sdf.select("serial_number")
        .unique()
        .sample(n=n_drives, seed=46)
        .get_column("serial_number")
    )

    pdf = (
        sdf.filter(pl.col("serial_number").is_in(sample_serials))
        .to_pandas()
    )

    pdf["serial_number"] = pdf["serial_number"].astype(str)
    pdf["first_date"] = pd.to_datetime(pdf["first_date"])
    pdf["last_date"] = pd.to_datetime(pdf["last_date"])
    pdf["event"] = pdf["event"].fillna(False).astype(bool)

    # Order drives by the beginning of their earliest episode.
    order = (
        pdf.groupby("serial_number", observed=True)["first_date"]
        .min()
        .sort_values()
        .index
        .tolist()
    )

    # Explicit categorical ordering for the y-axis.
    pdf["serial_number"] = pd.Categorical(
        pdf["serial_number"],
        categories=order,
        ordered=True,
    )

    pdf["episode_type"] = pdf["episode_id"].apply(
        lambda episode_id: "first" if episode_id == 0 else "later"
    )

    first_ep = pdf[pdf["episode_type"] == "first"]
    later_ep = pdf[pdf["episode_type"] == "later"]
    failed_ep = pdf[pdf["event"]]

    first_segments = hv.Segments(
        first_ep,
        kdims=[
            "first_date",
            "serial_number",
            "last_date",
            "serial_number",
        ],
        vdims=["event", "episode_id"],
        label="First episode",
    ).opts(
        color="event",
        cmap={True: "crimson", False: "steelblue"},
        line_width=6,
        tools=["hover"],
    )

    later_segments = hv.Segments(
        later_ep,
        kdims=[
            "first_date",
            "serial_number",
            "last_date",
            "serial_number",
        ],
        vdims=["event", "episode_id"],
        label="Later episode",
    ).opts(
        color="event",
        cmap={True: "crimson", False: "orange"},
        line_width=6,
        line_dash="dashed",
        tools=["hover"],
    )

    failure_markers = hv.Scatter(
        failed_ep,
        kdims=["last_date"],
        vdims=["serial_number", "episode_id"],
        label="Failure",
    ).opts(
        color="black",
        marker="x",
        size=9,
        line_width=2,
        tools=["hover"],
    )

    plot = (
        first_segments
        * later_segments
        * failure_markers
    ).opts(
        hv.opts.Overlay(
            height=max(500, n_drives * 10),
            width=950,
            xlabel="Date",
            ylabel="Serial number",
            title=(
                "Drive observation episodes — "
                "dashed = later episode, × = recorded failure"
            ),
            show_legend=True,
            legend_position="top_left",
        )
    )

    plot
    return


@app.cell
def _(survival_df):
    (
        survival_df.collect().hvplot.hist(y='duration_days')
    )
    return


@app.cell
def _(df, pl):
    df.group_by(['date', 'model']).agg(
        cnt=pl.col('serial_number').n_unique()
    ).sort('date').collect().hvplot.area(x='date', y='cnt', stacked=True, by='model').opts(frame_width=800, frame_height=500)
    return


@app.cell(disabled=True, hide_code=True)
def _(Path, gc, pl):
    SOURCE = "backblaze_drive_stats/*2024/*.parquet"
    OUTPUT_DIR = Path("smart_null_stats")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the source lazy. Do not use read_parquet(...).lazy().
    schema = pl.scan_parquet(SOURCE).collect_schema()

    smart_columns = [
        name for name in schema.names()
        if name.startswith("smart") and 'raw' in name
    ]

    for i, column in enumerate(smart_columns):
        output_path = OUTPUT_DIR / f"part_{i:04d}.parquet"

        query = (
            # Create a fresh scan so only these two columns are read.
            pl.scan_parquet(SOURCE, 
                            cache=False,
                            extra_columns="ignore"
                           )
            .select("model", column)
            .group_by("model")
            .agg(
                n_rows=pl.len(),
                n_nulls=pl.col(column).null_count(),
            )
            .with_columns(
                pl.lit(column).alias("smart_column"),
                (
                    pl.col("n_nulls") / pl.col("n_rows")
                ).alias("null_fraction"),
            )
            .select(
                "model",
                "smart_column",
                "n_rows",
                "n_nulls",
                "null_fraction",
            )
        )

        # Execute this column only and stream its result directly to disk.
        query.sink_parquet(
            output_path,
            engine="streaming",
            compression="zstd",
            maintain_order=False,
        )

        del query
        gc.collect()
    return


@app.cell
def _(relevant_smart):
    relevant_smart + ["date", "serial_number", "model", "failure"]
    return


@app.cell
def _(Path, pl):
    _names = [p.name for p in Path("./smart_null_stats").iterdir() if p.is_file()]
    _dfs = []
    for _name in _names:
        _df = pl.read_parquet(f"./smart_null_stats/{_name}")
        _dfs.append(
            _df
        )

    _df = pl.concat(_dfs, how='vertical').with_columns(
        occupancy = 1.0 - pl.col('null_fraction')
    )
    _df.hvplot.heatmap(
        x='model',
        y='smart_column',
        C='occupancy',
        cmap='viridis'
    ).opts(frame_width=1000, frame_height=1000,  xrotation=90)
    return


@app.cell
def _(df):
    df.select(['model', 'date', 'smart_9_raw']).collect().hvplot.hist(y='smart_9_raw')
    return


@app.cell
def _(spells):
    spells.collect()
    return


@app.cell
def _(df, np, pl):
    def fit_line(day_id, smart):
        xs, ys = [], []
        for x, y in zip(day_id, smart):
            if y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return {'slope': None, 'intercept': None}
        # reject any series that ever decreases
        if any(b < a for a, b in zip(ys, ys[1:])):
            return {'slope': None, 'intercept': None}
        slope, intercept = np.polyfit(xs, ys, 1)
        return {'slope': float(slope), 'intercept': float(intercept)}

    _result = (
        df.select(['date', 'model', 'serial_number', 'smart_9_raw', 'failure'])
        .remove(
            pl.col('failure').any().over('serial_number')
        )
        .sort('date')
        .group_by('serial_number', maintain_order=True)
        .agg(
            day_id=pl.int_range(pl.len()),
            smart=pl.col('smart_9_raw'),
            model=pl.col('model').first(),
        )
    ).collect()

    fits = _result.with_columns(
        pl.struct(['day_id', 'smart'])
        .map_elements(
            lambda s: fit_line(s['day_id'], s['smart']),
            return_dtype=pl.Struct({'slope': pl.Float64, 'intercept': pl.Float64}),
        )
        .alias('fit')
    ).unnest('fit')
    return (fits,)


@app.cell
def _(fits, pl):
    fits.filter(
        pl.col('slope').is_null()
    )
    return


@app.cell
def _(fits):
    fits.hvplot.hist(y='slope', bins=100, by='model', subplots=True)
    return


@app.cell
def _(pl):
    _x = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167]

    _y = [9096,9119,9143,9166,9190,9213,9237,9260,9284,9307,9331,9354,9378,9401,9425,9448,9472,9495,9519,9554,9577,9601,9624,9648,9671,9695,9718,9742,9765,9789,9812,9836,9859,9883,9906,9930,9953,9977,10000,10035,10059,10082,10106,10129,10152,10176,10199,10223,10246,10269,10293,10316,10340,10363,10387,10410,10434,10457,10492,10516,10539,10563,10586,10610,10634,10657,10669,10704,10727,10751,10774,10798,10821,10845,10868,10892,10915,10939,10962,10985,11009,11032,11056,11079,11103,11138,11161,11185,11208,11232,11255,11279,11279,0,31,47,78,93,109,140,156,187,219,234,266,281,313,328,359,375,406,422,453,469,500,516,547,579,594,626,642,673,689,720,736,767,783,814,845,861,893,908,940,955,987,1002,1034,1049,1081,1097,1128,1144,1175,1191,1222,1238,1269,1285,1316,1332,1363,1395,1410,1442,1457,1489,1504,1536,1551,1583,1598,1630,1645,1677,1692,1724,1755,1771]
    pl.DataFrame({'x':_x, 'y': _y}).hvplot.line(x='x', y='y')
    return


@app.cell
def _(pl, survival_df):
    _survival_df = survival_df.collect()
    (
        _survival_df.filter(
            pl.col('serial_number') == "10B0A0DKF97G"
        ).hvplot.scatter(x='first_date', y='episode_id')*
        _survival_df.filter(
            pl.col('serial_number') == "10B0A0DKF97G"
        ).hvplot.scatter(x='last_date', y='episode_id')
    )
    return


@app.cell(hide_code=True)
def _(df, hv, pl):
    from scipy.stats import beta
    import numpy as np


    alpha = 0.05

    def jeffreys_interval(k, n, alpha=0.01):
        """Jeffreys interval for a binomial proportion k/n."""
        lower = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        upper = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        lower = np.where(k == 0, 0.0, lower)
        upper = np.where(k == n, 1.0, upper)
        return lower, upper

    agg = (
        df.select(['date', 'serial_number', 'model', 'failure'])
        .group_by('model')
        .agg(
            failure_count=pl.col('failure').sum(),
            total_count=pl.len(),
        )
        .collect()
    )

    k = agg['failure_count'].to_numpy()
    n = agg['total_count'].to_numpy()

    p_lo, p_hi = jeffreys_interval(k, n, alpha=alpha)

    agg = agg.with_columns([
        (100.0 - 100.0 * pl.col('failure_count') / pl.col('total_count')).alias('failure_rate_pct'),
        pl.Series('ci_lower', 100.0 * (1 - p_hi)),
        pl.Series('ci_upper', 100.0 * (1 - p_lo)),
    ])

    agg = agg.with_columns([
        (pl.col('failure_rate_pct') - pl.col('ci_lower')).alias('err_neg'),
        (pl.col('ci_upper') - pl.col('failure_rate_pct')).alias('err_pos'),
    ])

    # sort so both panels share the same model order on the x-axis
    agg = agg.sort('total_count', descending=True)
    _pdf = agg.to_pandas()
    model_order = _pdf['model'].tolist()

    # --- main panel: scatter + error bars ---
    scatter = agg.hvplot.scatter(
        x='model', y='failure_rate_pct', size=100
    ).opts(
        frame_width=700, frame_height=350,
        xlim=(None, None),
        xrotation=45,
        ylabel='Failure rate (%)',
    )

    errorbars = hv.ErrorBars(
        _pdf, kdims='model', vdims=['failure_rate_pct', 'err_neg', 'err_pos']
    )

    main = (scatter * errorbars).opts(
        hv.opts.Scatter(color='#2b6cb0'),
        hv.opts.ErrorBars(line_width=1.5, color='#2b6cb0'),
    )
    main = main.redim.values(model=model_order)

    # --- volume panel: sample size per model ---
    volume = hv.Bars(_pdf, kdims='model', vdims='total_count').opts(
        frame_width=700, frame_height=100,
        color='#a0aec0', line_color=None, alpha=0.7,
        ylabel='n', xlabel='',
        xaxis=None,  # hide duplicate x-axis, top panel shares labels via layout
        tools=['hover'],
    )
    volume = volume.redim.values(model=model_order)

    layout = (volume + main).cols(1).opts(
        hv.opts.Layout(shared_axes=True)
    )

    layout
    return (np,)


@app.cell
def _(pl, units):
    from lifelines import KaplanMeierFitter

    _units = units.filter(
        pl.col('status').is_not_null(),
        pl.col('exit_poh').is_not_null(),
        pl.col('duration').is_not_null()
    ).with_columns(
        pl.when(pl.col('status') == 'died').then(pl.lit(True)).otherwise(pl.lit(False)).alias('status')
    ).select(['duration', 'status', 'exit_poh']).collect()

    kmf_1 = KaplanMeierFitter().fit(
        durations=_units['exit_poh'].to_numpy(),
        event_observed=_units['status'].to_numpy()
    )
    kmf_2 = KaplanMeierFitter().fit(
        durations=_units['duration'].to_numpy(),
        event_observed=_units['status'].to_numpy()
    )
    return kmf_1, kmf_2


@app.cell
def _(kmf_1, kmf_2, pl):
    (
        pl.DataFrame(kmf_1.survival_function_).with_columns(
            timeline=pl.row_index()/24
        ).hvplot.line(x='timeline', y='KM_estimate')+
        pl.DataFrame(kmf_2.survival_function_).with_columns(
            timeline=pl.row_index()
        ).hvplot.line(x='timeline', y='KM_estimate')
    ).opts(shared_axes=False)
    return


@app.cell
def _(units):
    units.columns
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
