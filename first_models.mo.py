import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")


@app.cell(hide_code=True)
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
    from lifelines import KaplanMeierFitter


    GAP_TOLERANCE = 7    
    MAX_BRIDGED   = 5    
    MAX_SPELLS    = 10    
    return (
        GAP_TOLERANCE,
        KaplanMeierFitter,
        MAX_BRIDGED,
        MAX_SPELLS,
        glob,
        hv,
        mo,
        pl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read and prep the data
    """)
    return


@app.cell(hide_code=True)
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
    return df, df_topk


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
            vault_id=pl.col("vault_id").sort_by("date").last(),   # <-- add this
            n_bridged_gaps=pl.col("bridged_gap").sum(),

            # SMART 9 (power-on hours) entry/exit within this spell.
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
    return spells, units


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## KM estimator
    """)
    return


@app.cell
def units_coll(units):
    units_coll = units.collect()
    return


@app.cell(hide_code=True)
def _(KaplanMeierFitter, hv, pl):
    def fit_km_by_group(
        df: pl.DataFrame,
        group_col: str,
        duration_col: str,
        event_col: str,
        kind: str = "duration_entry",
        entry_col: str = None,
        groups: list = None,
        alpha: float = 0.05,
        time_divisor: float = None,
        ci_alpha_pct: int = None,
        show_interval: bool = True,   # now just controls INITIAL visibility, band is always built
        ci_area_alpha: float = 0.15,
    ):
        """Fit KM per group.
        kind="duration":       KaplanMeierFitter.fit(durations, event_observed)
        kind="duration_entry": KaplanMeierFitter.fit(durations, event_observed, entry=entry)

        The CI band is always computed and added to the overlay as its own
        legend entry ("... — CI"), toggleable by clicking the legend in the
        rendered/exported Bokeh HTML (click_policy='mute').
        """
        assert kind in ("duration", "duration_entry")
        if kind == "duration_entry" and entry_col is None:
            raise ValueError("entry_col is required when kind='duration_entry'")
        groups = groups if groups is not None else df[group_col].unique().to_list()
        ci_alpha_pct = ci_alpha_pct or int((1 - alpha) * 100)
        fitters, tables, layers = {}, {}, {}

        for g in sorted(groups):
            sub = df.filter(pl.col(group_col) == g)
            valid = pl.col(duration_col).is_not_null()
            if kind == "duration_entry":
                valid = valid & pl.col(entry_col).is_not_null() & (pl.col(duration_col) > pl.col(entry_col))
            sub = sub.filter(valid)
            if sub.is_empty():
                continue
            if time_divisor is not None:
                cols = [(pl.col(duration_col) / time_divisor).alias(duration_col)]
                if kind == "duration_entry":
                    cols.append((pl.col(entry_col) / time_divisor).alias(entry_col))
                sub = sub.with_columns(cols)

            kmf = KaplanMeierFitter(alpha=alpha)
            kmf.fit(
                durations=sub[duration_col].to_numpy(),
                event_observed=sub[event_col].to_numpy(),
                entry=sub[entry_col].to_numpy() if kind == "duration_entry" else None,
                label="KM_estimate",
            )
            fitters[g] = kmf

            surv_df = kmf.survival_function_.join(kmf.confidence_interval_survival_function_).reset_index()
            surv_pl = pl.from_pandas(surv_df)
            surv_pl.columns = ["days", "KM_estimate", "KM_lower", "KM_higher"]
            tables[g] = surv_pl

            band = surv_pl.hvplot.area(
                x="days", y="KM_lower", y2="KM_higher",
                alpha=ci_area_alpha, label=f"{g} — {ci_alpha_pct}% CI",
            ).opts(
                muted=not show_interval,   # start hidden unless show_interval=True
                muted_alpha=0,             # fully invisible when muted (not just faded)
            )

            line = surv_pl.hvplot.line(
                x="days", y="KM_estimate", line_width=2, label=str(g),
            ).opts(
                muted_alpha=0.15,          # if someone mutes the line itself, fade instead of vanish
            )

            # keep both as separate, independently-togglable legend entries
            layers[g] = band * line

        return fitters, tables, layers


    def km_overlay(
        layers, groups,
        title="Kaplan–Meier survival curves", xlabel="Days",
        width=900, height=550, ylim=(0.8, 1.05),
    ):
        ov = hv.Overlay(
            [layers[g] for g in groups if g in layers]
        ).opts(
            title=title, xlabel=xlabel, ylabel="Survival probability",
            ylim=ylim, width=width, height=height,
            legend_position="bottom", legend_cols=2,
            # ensures bokeh legend behaves as "click to mute/unmute" in the exported HTML
            legend_opts={"click_policy": "mute"},
        )
        return ov

    return fit_km_by_group, km_overlay


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plotting main KM
    | model | cnt |
    |---|---|
    | WDC WUH722222ALE6L4 | 44725 |
    | TOSHIBA MG08ACA16TA | 41056 |
    | TOSHIBA MG07ACA14TA | 38952 |
    | ST16000NM001G | 35182 |
    | WDC WUH721816ALE6L4 | 26646 |
    | ST12000NM0008 | 20649 |
    | ST8000NM0055 | 14833 |
    | HGST HUH721212ALE604 | 14317 |
    | ST12000NM001G | 13790 |
    | ST14000NM001G | 11169 |
    """)
    return


@app.cell
def _(spells):
    spells_coll = spells.collect()
    return (spells_coll,)


@app.cell
def km_estimators(df_topk, fit_km_by_group, km_overlay, mo, spells_coll):
    from holoviews import save
    _km_groups = sorted(
        df_topk.top_k(k=5, by='count')['model'].to_list()
    )

    _fitters_days, _tables_days, _layers_days = fit_km_by_group(
        spells_coll, "model", "observed_days", "event",
        kind="duration",
        time_divisor=1.0, alpha=0.01,
    )
    _km_days = km_overlay(
        _layers_days, _km_groups,
        title="KM — Observed Days (spells)", xlabel="Days",
        ylim=(0.8, 1.01), 
        width=600,height=400
    )

    _fitters_hours, _tables_hours, _layers_hours = fit_km_by_group(
        spells_coll, "model", "exit_hours", "event",
        kind="duration_entry", entry_col="entry_hours",
        time_divisor=24.0, alpha=0.01,
    )
    _km_hours = km_overlay(
        _layers_hours, _km_groups,
        title="KM — Power-On Hours (spells)", xlabel="Days (hours / 24)",
        ylim=(0.8, 1.01),
        width=600,height=400
    )

    save(_km_days+_km_hours, './backblaze_drive_stats/results/estimators/km_days_hours.html')
    mo.sidebar(mo.vstack([_km_days, _km_hours]), width=900)
    #mo.vstack([_km_days, _km_hours])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Visualizing migration
    """)
    return


@app.cell
def _(spells):
    spells_c = spells.collect()
    return (spells_c,)


@app.cell
def _(pl, spells_c):
    from datetime import date
    spells_c_extra = spells_c.with_columns(
        exit_type=(
            pl.when(pl.col('event')).then(pl.lit('failed'))
            .when(pl.col('last_date') >= date(2025, 12, 25)).then(pl.lit('administrative'))
            .otherwise(pl.lit('unresolved'))
        ),
        exit_month=pl.col('last_date').dt.truncate('1mo')
    )
    return (spells_c_extra,)


@app.cell
def _(pl, spells_c_extra):
    (
        spells_c_extra
        .group_by(['vault_id', 'exit_month'])
        .agg(
            cnt_unresolved=(pl.col('exit_type')=='unresolved').sum(),
            cnt_total=pl.len()
        ).with_columns(
            ratio=(pl.col('cnt_unresolved'))/(pl.col('cnt_total'))
        )
        #.with_columns(
        #    pl.col('vault_id').cast(pl.String),
        #    pl.col('exit_month').cast(pl.String),
        #)
        .hvplot.heatmap(
            x='vault_id',
            y='exit_month',
            C='ratio',
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Identifying migration
    """)
    return


@app.cell
def _(pl, spells_c_extra):
    month_spine = spells_c_extra.select(
        pl.date_range(spells_c_extra['first_date'].min(), spells_c_extra['last_date'].max(), '1mo', eager=True).alias('month')
    )
    return (month_spine,)


@app.cell(hide_code=True)
def _(month_spine, pl, spells_c_extra):
    active = (
        spells_c_extra.join(month_spine, how='cross')
        .filter(
            (pl.col('first_date') <= pl.col('month').dt.month_end())
            & (pl.col('last_date') >= pl.col('month'))
        )
        .group_by(['vault_id', 'month'])
        .agg(n_active=pl.n_unique('serial_number'))
    )


    exits = (
        spells_c_extra
        .filter(pl.col('exit_type') == 'unresolved')
        .group_by(['vault_id', 'exit_month'])
        .agg(n_unresolved=pl.len())
        .rename({'exit_month': 'month'})
    )


    vault_month = (
        active.join(exits, on=['vault_id', 'month'], how='left')
        .with_columns(n_unresolved=pl.col('n_unresolved').fill_null(0))
        .with_columns(ratio=pl.col('n_unresolved') / pl.col('n_active'))
        .with_columns(is_migration_event=pl.col('n_unresolved')>20)
    )


    [
        vault_month.hvplot.heatmap(
            x='month', y='vault_id', C='ratio'
        ),
        vault_month.hvplot.hist(
            y='ratio'
        )
    ]
    return (vault_month,)


@app.cell
def _(pl, spells_c_extra, vault_month):
    spells_with_migrated = (
        spells_c_extra
        .join(
            vault_month.select('vault_id', 'month', 'is_migration_event'),
            left_on=['vault_id', 'exit_month'],
            right_on=['vault_id', 'month'],
            how='left',
        )
        .with_columns(
            exit_type=pl.when(
                (pl.col('exit_type') == 'unresolved') & pl.col('is_migration_event')
            ).then(pl.lit('migration'))
            .when(pl.col('exit_type') == 'unresolved')
            .then(pl.lit('unknown_dropout'))
            .otherwise(pl.col('exit_type'))
        )
        .with_columns(
            pl.when(
                pl.col('is_migration_event') & pl.col('event')
            ).then(pl.lit(False)).otherwise(
                pl.col('is_migration_event')
            ).alias('is_migration_event')
        )
    )
    _code_map = {'failed': 1, 'migration': 2, 'unknown_dropout': 1, 'administrative': 0}
    spells_with_migrated = spells_with_migrated.with_columns(
        exit_type_code = pl.col('exit_type').replace(_code_map).cast(pl.Int8)
    )
    spells_with_migrated = spells_with_migrated.remove(
        pl.col('entry_hours').is_null() | (
            pl.col('entry_hours') >= pl.col('exit_hours')
        )
    )
    return (spells_with_migrated,)


@app.cell
def _(pl, spells_with_migrated):
    spells_with_migrated.filter(

            pl.col('entry_hours') >= pl.col('exit_hours')
    )
    return


@app.cell
def _(pl, spells_with_migrated):
    spells_with_migrated.filter(
        pl.col('is_migration_event')
    ).group_by('model').agg(cnt=pl.len())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Aalen Johansen
    """)
    return


@app.cell
def _(KaplanMeierFitter, spells_with_migrated):
    from lifelines import AalenJohansenFitter

    ajf_fail = AalenJohansenFitter()
    ajf_fail.fit(
        durations=spells_with_migrated['exit_hours'], 
        event_observed=spells_with_migrated['exit_type_code'],
        event_of_interest=1,
        entry=spells_with_migrated['entry_hours']
    )

    ajf_mig = AalenJohansenFitter()
    ajf_mig.fit(
        durations=spells_with_migrated['exit_hours'], 
        event_observed=spells_with_migrated['exit_type_code'],
        event_of_interest=2,
        entry=spells_with_migrated['entry_hours']
    )

    km_fail = KaplanMeierFitter()
    km_fail.fit(
        durations=spells_with_migrated['exit_hours'],
        event_observed=( spells_with_migrated['exit_type_code'] > 0),
        entry=spells_with_migrated['entry_hours']
    )
    return ajf_fail, ajf_mig, km_fail


@app.cell
def _(ajf_fail, ajf_mig, hv, km_fail, pl):
    from bokeh.io import show
    from bokeh.io import save as save_bk
    def sum_step_curves(dfs, names=None, times=None, fill_before=0.0):
        if names is None:
            names = [f"curve_{i}" for i in range(len(dfs))]
        norm = []
        for df, n in zip(dfs, names):
            if not isinstance(df, pl.DataFrame):
                df = pl.from_pandas(df)
            t, v = df.columns[0], df.columns[1]
            norm.append(
                df.select(pl.col(t).cast(pl.Int64).alias("time"),
                          pl.col(v).cast(pl.Float64).alias(n))
                  .sort("time")
            )
        if times is None:
            grid = pl.concat([d.select("time") for d in norm]).unique().sort("time")
        else:
            grid = pl.DataFrame({"time": pl.Series(times, dtype=pl.Int64)}).sort("time")
        out = grid
        for d in norm:
            out = out.join_asof(d, on="time", strategy="backward")
        return (
            out.with_columns(pl.col(names).fill_null(fill_before))
               .with_columns(total=pl.sum_horizontal(names))
        )
    res = sum_step_curves(
        [
       pl.from_pandas(ajf_fail.cumulative_density_.reset_index().rename({'event_at': 'time', 'CIF_1': 'ajf_fail'})),
        pl.from_pandas(ajf_mig.cumulative_density_.reset_index().rename({'event_at': 'time', 'CIF_2': 'ajf_mig'})),
        pl.from_pandas(km_fail.cumulative_density_.reset_index().rename({'timeline': 'time', 'KM_estimate': 'km_fail'})),
        ],
        names=["ajf_fail", "ajf_mig", "km_fail"],
    )
    def latex_labels(plot, element):
        plot.state.yaxis.axis_label = r"$$\hat{S}(t)$$"
    overlay = (
        pl.from_pandas(ajf_fail.cumulative_density_.reset_index())
          .with_columns(survival=1.0 - pl.col('CIF_1'))
          .hvplot.line(y='survival', x='event_at', label='AJ fail')
        * pl.from_pandas(ajf_mig.cumulative_density_.reset_index())
            .with_columns(survival=1.0 - pl.col('CIF_2'))
            .hvplot.line(y='survival', x='event_at', label='AJ migration')
        * pl.from_pandas(km_fail.cumulative_density_.reset_index())
            .with_columns(survival=1.0 - pl.col('KM_estimate'))
            .hvplot.line(y='survival', x='timeline', label='KM')
    ).opts(
        xlabel='hours',
        hooks=[latex_labels],
    )
    fig = hv.render(overlay)          # -> real bokeh Figure
    fig.yaxis.axis_label = r"$$\hat{S}(t)$$"
    save_bk(fig, filename='./backblaze_drive_stats/results/estimators/aj_fail_migration_km.html')
    show(fig)
    return


app._unparsable_cell(
    r"""
    ajf_fail.
    """,
    name="_"
)


@app.cell
def _(ajf_fail, ajf_mig, km_fail, pl):

    #for t in km_fail.survival_function_.reset_index()['timeline'].to_list():
    _t = km_fail.survival_function_.reset_index()['timeline'].to_list()
    total = (
        km_fail.predict(_t)
        + ajf_fail.predict(_t)
        + ajf_mig.predict(_t)
    )

    pl.from_pandas(total.reset_index()).hvplot.scatter(x='index', y='0')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Perform logrank
    """)
    return


@app.cell
def _(ajf_fail, km_fail, pl):
    _kmfail_df = pl.from_pandas(km_fail.cumulative_density_.reset_index()).with_columns(survival=1.0 - pl.col('KM_estimate')).rename({'timeline': 'time', 'survival': 's'})
    _ajfail_df = pl.from_pandas(ajf_fail.cumulative_density_.reset_index()).with_columns(survival=1.0 - pl.col('CIF_1')).rename({'event_at': 'time', 'survival': 's'})
    [
        _kmfail_df,
        _ajfail_df
    ]
    from lifelines.statistics import logrank_test

    _durations_km = km_fail.durations
    _events_km    = km_fail.event_observed

    _durations_ajf = ajf_fail.durations
    _events_ajf    = ajf_fail.event_observed   # 1 = fail, 0 = censored/other

    _result = logrank_test(
        _durations_km, _durations_ajf,
        event_observed_A=_events_km,
        event_observed_B=_events_ajf,
    )

    _result.summary
    return


@app.cell
def _(km_fail):
    km_fail.survival_function_
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
