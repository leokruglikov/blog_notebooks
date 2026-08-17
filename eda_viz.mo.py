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
    from bokeh.models import HoverTool
    import math


    GAP_TOLERANCE = 5    
    MAX_BRIDGED   = 3    
    MAX_SPELLS    = 2    
    return (
        GAP_TOLERANCE,
        MAX_BRIDGED,
        MAX_SPELLS,
        Path,
        gc,
        glob,
        hv,
        math,
        mo,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Reading & working with the data
    """)
    return


@app.cell
def _(glob, pl):
    lfs = [
        pl.scan_parquet(f, hive_partitioning=True)
        for f in glob.glob(
            "backblaze_drive_stats/**/*.parquet",
            recursive=True,
        )
    ]
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
        pl.col('model').is_in(df_topk.top_k(15, by='count')['model'])
    )
    #df.group_by('serial_number').agg(cnt=pl.len()).collect()
    #df = df.remove(
    #    pl.col('serial_number').len().over('serial_number').is_in([365, 366, 364])
    #)
    #df.group_by('serial_number').agg(cnt=pl.len()).collect()
    return (df,)


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
        # Exclude observations after the first failure.
        .filter(pl.col("date") <= pl.col("last_date"))
        .group_by("serial_number")
        .agg(
            observed_days=pl.len(),

            smart9_observed_rows=(
                pl.col("smart_9_raw")
                .is_not_null()
                .sum()
            ),

            # daily is already ordered by serial_number and date
            entry_hours=(
                pl.col("smart_9_raw")
                .filter(pl.col("smart_9_raw").is_not_null())
                .first()
            ),

            exit_hours=(
                pl.col("smart_9_raw")
                .filter(pl.col("smart_9_raw").is_not_null())
                .last()
            ),

            smart9_first_date=(
                pl.col("date")
                .filter(pl.col("smart_9_raw").is_not_null())
                .first()
            ),

            smart9_last_date=(
                pl.col("date")
                .filter(pl.col("smart_9_raw").is_not_null())
                .last()
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
    return daily, spells, units


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fleet by models vs time
    """)
    return


@app.cell
def _(df, math, pl):
    lf = (df.group_by(['date', 'model'])
            .agg(cnt=pl.col('serial_number').n_unique())
            .sort('date')
            .collect())

    n_models = lf['model'].n_unique()

    plot = (lf.hvplot.area(
                x='date', y='cnt', by='model', stacked=True,
                alpha=0.9, line_width=0.5, legend='bottom',
                xlabel='', ylabel='Drives in service',
                title='Fleet size by model', grid=True,
            )
            .opts(frame_width=700, frame_height=350,
                  fontsize={'title': 15, 'labels': 12, 'ticks': 10, 'legend': 9},
                  toolbar='above', active_tools=[],
                  legend_position='bottom',
                  legend_cols=math.ceil(n_models / 3)))
    return (plot,)


@app.cell
def _(plot, save):
    #from holoviews import save
    save(plot, filename='./backblaze_drive_stats/results/eda/area_plot_fleet_size.html')
    plot
    return


@app.cell(disabled=True, hide_code=True)
def _(Path, gc, pl):
    from tqdm import tqdm
    SOURCE = "backblaze_drive_stats/**/*.parquet"
    OUTPUT_DIR = Path("smart_null_stats")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the source lazy. Do not use read_parquet(...).lazy().
    schema = pl.scan_parquet(SOURCE).collect_schema()

    smart_columns = [
        name for name in schema.names()
        if name.startswith("smart") and 'raw' in name
    ]

    i=0
    for column in tqdm(smart_columns):
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
        i+=1
    return


@app.cell
def _(Path, pl):
    from holoviews import save
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
    _heatmap = _df.hvplot.heatmap(
        x='model',
        y='smart_column',
        C='occupancy',
        cmap='viridis'
    ).opts(frame_width=1000, frame_height=1000,  xrotation=90)
    #save(_heatmap, "./backblaze_drive_stats/results/eda/heatmap_smart_null_rate.html")
    return (save,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploring units
    """)
    return


@app.cell
def _(spells):
    spells.collect()
    return


@app.cell
def _(units):
    units.collect()
    return


@app.cell
def _():
    # 10A0A0AHF97G
    # ZA1478Y4 - failed but then reappeared and flagged as censored instead of failed.
    # ZL0050Z5 - flaky
    return


@app.cell
def _(df):
    df.select(['date', 'serial_number', 'model', 'failure', 'cluster_id', 'vault_id', 'pod_slot_num', 'smart_9_raw']).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Gantt-like
    """)
    return


@app.cell
def _(pl, spells, units):
    def _sample_each(df: pl.LazyFrame, n=4, which={'flaky': (True, 2), 'resurrected': (True, 3), 'interrupted': (True, 3), 'clean': (True,1)}, seed=43):
        serials = []
        serials_dic = {}
        for c in which.keys():
            serials_ = df.filter(
                pl.col(c) == which[c][0]
            ).collect().sample(n=which[c][1], seed=seed)['serial_number'].to_list()
            serials.extend(serials_)
            serials_dic[c]=serials_
        return serials, serials_dic

    serials, serials_dic = _sample_each(units)


    serial_to_mode = {
        serial: mode
        for mode, serial_list in serials_dic.items()
        for serial in serial_list
    }

    plot_df = (
        spells
        .filter(pl.col("serial_number").is_in(serials))
        .sort(["serial_number", "first_date"])
        .with_columns(
            end_date=pl.col("last_date") + pl.duration(days=1),
            state=pl.when(pl.col("event"))
                .then(pl.lit("failed"))
                .otherwise(pl.lit("observed")),
        )
        .collect()
        .to_pandas()
    )

    # New label combining mode + serial number
    plot_df["mode"] = plot_df["serial_number"].map(serial_to_mode)
    plot_df["label"] = plot_df["mode"] + " : " + plot_df["serial_number"].astype(str)

    # Order labels by mode first, then serial number, so groups cluster together
    order = (
        plot_df[["mode", "serial_number", "label"]]
        .drop_duplicates()
        .sort_values(["mode", "serial_number"])["label"]
        .tolist()
    )
    y_map = {label: i for i, label in enumerate(order)}
    plot_df["y0"] = plot_df["label"].map(y_map) - 0.35
    plot_df["y1"] = plot_df["label"].map(y_map) + 0.35
    return order, plot_df, serials_dic, y_map


@app.cell
def _(hv, order, plot_df, y_map):
    gantt_ = hv.Rectangles(
        plot_df,
        kdims=["first_date", "y0", "end_date", "y1"],
        vdims=["label", "state", "spell_id"],
    ).opts(
        width=900,
        height=max(250, 45 * len(order)),
        color="state",
        cmap={"observed": "#4C78A8", "failed": "#E45756"},
        yticks=[(i, label) for label, i in y_map.items()],
        tools=["hover"],
        xlabel="Date",
        ylabel="Serial number (mode)",
        title="Drive observation spells by mode",
    )

    #save(gantt_, 'backblaze_drive_stats/results/eda/gantt_plot_sampled.html')
    gantt_
    return


@app.cell
def _(serials_dic):
    serials_dic
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fits & outliers
    """)
    return


@app.cell
def _(df, pl):
    import numpy as np

    def fit_line(day_id, smart):
        xs, ys = [], []
        for x, y in zip(day_id, smart):
            if y is not None:
                xs.append(x)
                ys.append(y)
        if len(xs) < 2:
            return {'slope': None, 'intercept': None}
        ## reject any series that ever decreases
        #if any(b < a for a, b in zip(ys, ys[1:])):
        #    return {'slope': None, 'intercept': None}
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
    return fits, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    trying to find correlation between negative /outliers slope and the

    observation: only seagate (ST) and (almost) all 365 days
    """)
    return


@app.cell
def _(fits, units):
    joined_fits =  fits.join(
        other=units.collect(),
        on='serial_number',
        how='inner'
    )
    return (joined_fits,)


@app.cell
def _(joined_fits, pl):
    #(
    #    fits.with_columns(
    #        slope_sign=pl.when(pl.col('slope')>0).then(pl.lit("+")).otherwise(pl.lit("-"))
    #    ).group_by('slope_sign').agg(
    #        cnt=pl.len()
    #    )
    #).hvplot.bar(x='slope_sign', y='cnt')
    (
        joined_fits.with_columns(
            pl.col('clean').cast(pl.String),
            pl.col('interrupted').cast(pl.String),
        ).hvplot.scatter(
            x='slope',
            y='interrupted'
        )
    )
    return


@app.cell
def _(fits, units):
    fits.join(
        other=units.collect(),
        on='serial_number',
        how='inner'
    )
    return


@app.cell
def _(hv, joined_fits, pl):
    sampled = joined_fits.filter(
        ~(pl.col('slope').is_between(23, 35))
    ).sample(10, seed=42)

    plots = [
        hv.Scatter(
            (row['day_id'], row['smart']),
            kdims='day_id', vdims='smart'
        ).opts(
            title=f"{row['serial_number']} (slope={row['slope']:.2f})",
            width=250, height=200,
        )
        for row in sampled.iter_rows(named=True)
    ]

    hv.Layout(plots).cols(5)
    return (sampled,)


@app.cell
def _(sampled):
    sampled
    return


@app.cell
def _(joined_fits, pl):
    joined_fits.filter(
        ~(pl.col('slope').is_between(23, 25))
    ).hvplot.hist(y='slope', bins=100)
    return


@app.cell
def _(daily):
    daily.select([
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "failure",
    "gap_days",
    "prev_failure",
    "bridged_gap",
    "breaks_spell",
    "spell_id"
    ]).collect()
    return


@app.cell
def _(daily, pl):
    daily.select(['date', 'serial_number', 'smart_9_raw', 'model']).sort('date').filter(
        ( pl.col('smart_9_raw').diff() < 0 ).any().over('serial_number')
    ).collect()
    return


@app.cell
def _(df, pl):
    (
        df.select(['date', 'smart_9_raw', 'model', 'serial_number', 'vault_id']).group_by('serial_number')
            .agg(n_unique=pl.n_unique('vault_id'))
    ).collect()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Estimating rates
    """)
    return


@app.cell
def _(df, hv, np, pl, save):
    from scipy.stats import beta, chi2
    alpha = 0.04
    by_what = 'model'

    def jeffreys_interval(k, n, alpha=0.01):
        """Jeffreys interval for a binomial proportion k/n."""
        lower = beta.ppf(alpha / 2, k + 0.5, n - k + 0.5)
        upper = beta.ppf(1 - alpha / 2, k + 0.5, n - k + 0.5)
        lower = np.where(k == 0, 0.0, lower)
        upper = np.where(k == n, 1.0, upper)
        return lower, upper


    agg = (
        df.select(["date", "serial_number", "model", "failure", 'cluster_id', 'datacenter'])
        .with_columns(
            pl.col(by_what).cast(pl.String).alias(by_what)
        )
        .group_by(by_what)
        .agg(
            failure_count=pl.col("failure").sum(),
            drive_days=pl.len(),
        )
        .collect()
    )

    k = agg["failure_count"].to_numpy()
    exposure_years = agg["drive_days"].to_numpy() / 365.25

    alpha = 0.05

    count_lo = np.where(
        k == 0,
        0.0,
        0.5 * chi2.ppf(alpha / 2, 2 * k),
    )

    count_hi = 0.5 * chi2.ppf(
        1 - alpha / 2,
        2 * (k + 1),
    )

    afr = 100.0 * k / exposure_years
    afr_lo = 100.0 * count_lo / exposure_years
    afr_hi = 100.0 * count_hi / exposure_years

    agg = agg.with_columns([
        pl.Series("afr_pct", afr),
        pl.Series("ci_lower", afr_lo),
        pl.Series("ci_upper", afr_hi),
    ]).with_columns([
        (pl.col("afr_pct") - pl.col("ci_lower")).alias("err_neg"),
        (pl.col("ci_upper") - pl.col("afr_pct")).alias("err_pos"),
    ])

    # sort so both panels share the same model order on the x-axis
    _pdf = agg.to_pandas()
    model_order = _pdf[by_what].tolist()

    # --- main panel: scatter + error bars ---
    scatter = agg.hvplot.scatter(
        x=by_what, y='afr_pct', size=100
    ).opts(
        frame_width=700, frame_height=350,
        xlim=(None, None),
        xrotation=45,
        ylabel='Failure rate (%)',
        show_grid=True
    )

    errorbars = hv.ErrorBars(
        _pdf, kdims=by_what, vdims=['afr_pct', 'err_neg', 'err_pos']
    )


    def style_errorbar_caps(plot, element):
        whisker = plot.handles["glyph"]

        for cap in (whisker.upper_head, whisker.lower_head):
            cap.line_color = "#e87b8f"
            cap.line_width = 2.5


    main = (scatter * errorbars).opts(
        hv.opts.Scatter(color="red"),
        hv.opts.ErrorBars(
            line_width=2.5,  # thickness of the error-bar stems
            color="#e87b8f",
            hooks=[style_errorbar_caps],
        ),
    )

    main = main.redim.values(model=model_order)

    volume = hv.Bars(_pdf, kdims=by_what, vdims='drive_days').opts(
        frame_width=700, frame_height=100,
        color='#0084ff', line_color=None, alpha=0.7,
        ylabel='n', xlabel='',
        xaxis=None,  # hide duplicate x-axis, top panel shares labels via layout
        tools=['hover'],
    )
    volume = volume.redim.values(model=model_order)

    layout = (volume + main).cols(1).opts(
        hv.opts.Layout(shared_axes=True)
    )

    save(layout, f'./backblaze_drive_stats/results/eda/failure_rates_errorbars_{by_what}.html')
    layout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
 
    """)
    return


@app.cell
def _(pl, units):
    units.with_columns(
        month=pl.col('first_date').dt.month()
    ).collect()
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
