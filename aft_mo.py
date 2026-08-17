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
    from lifelines import KaplanMeierFitter

    GAP_TOLERANCE = 7    
    MAX_BRIDGED   = 5    
    MAX_SPELLS    = 10    
    return GAP_TOLERANCE, MAX_BRIDGED, MAX_SPELLS, glob, hv, mo, pl


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


@app.cell
def _(spells, units):
    units_c = units.collect()
    spells_c = spells.collect()
    return (units_c,)


@app.cell
def _(units_c):
    units_c
    return


@app.cell
def _(df, pl, units_c):
    import lifelines

    results = []
    _models = df.group_by('model').agg(cnt=pl.len()).sort('cnt').top_k(by='cnt', k=10).collect()['model']
    for model_name, grp in units_c.filter(pl.col('exit_hours').is_not_nan()).group_by("model"):
        if model_name[0] not in _models:
            continue
        d = grp["exit_hours"].to_numpy() + 1
        e = grp["event"].to_numpy()
        t0 = grp["entry_hours"].to_numpy()

        wf = lifelines.WeibullFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)
        lf = lifelines.LogNormalFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)
        llf = lifelines.LogLogisticFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)


        results.append({
            "model": model_name[0],
            "n": grp.height, "events": int(e.sum()),
            "weibull_rho": wf.rho_, "weibull_rho_lo": wf.confidence_interval_.iloc[0,0],
            "weibull_rho_hi": wf.confidence_interval_.iloc[0,1],
            "weibull_lambda": wf.lambda_,
            "weibull_aic": wf.AIC_,
            "lognormal_aic": lf.AIC_,
            "loglogistic_aic": lf.AIC_
        })

    beta_table = pl.DataFrame(results).sort("model")
    return beta_table, lifelines


@app.cell
def _(beta_table, pl):
    beta_table.with_columns(
        ratio=100.0*pl.col('events')/pl.col('n')
    )
    return


@app.cell
def _(beta_table):
    (
        beta_table.hvplot.scatter(x='model', y='weibull_rho')*
        beta_table.hvplot.errorbars(x='model', y='weibull_rho', yerr1='weibull_rho_lo', yerr2='weibull_rho_hi')
    ).opts(xrotation=40)
    return


@app.cell(hide_code=True)
def _(beta_table, df, hv, lifelines, pl, units_c):
    import numpy as np
    from functools import reduce

    _times = np.linspace(0.0, 2500.0, 10000) * 24

    _models = (
        df.group_by("model")
          .agg(cnt=pl.len())
          .sort("cnt")
          .top_k(by="cnt", k=10)
          .collect()["model"]
    )

    _colors = hv.Cycle().values
    _plots = []

    for _i, _model_name in enumerate(_models):

        _color = _colors[_i % len(_colors)]

        # Weibull fitted curve
        _row = beta_table.filter(pl.col("model") == _model_name)

        _lambda = _row["weibull_lambda"].item()
        _rho = _row["weibull_rho"].item()

        _weibull = hv.Curve(
            (
                _times,
                np.exp(-(_times / _lambda) ** _rho),
            ),
            label=f"{_model_name} Weibull",
        ).opts(
            color=_color,
            line_width=5,
        )

        # Kaplan-Meier curve
        _grp = units_c.filter(
            (pl.col("model") == _model_name)
            & pl.col("exit_hours").is_not_nan() & 
            ( pl.col("exit_hours") > pl.col("entry_hours"))
        )

        _d = _grp["exit_hours"].to_numpy() + 1
        _e = _grp["event"].to_numpy()
        _t0 = _grp["entry_hours"].to_numpy()

        _kmf = lifelines.KaplanMeierFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
        )

        _km = hv.Curve(
            (
                _kmf.survival_function_.index.to_numpy(),
                _kmf.survival_function_.iloc[:, 0].to_numpy(),
            ),
            label=f"{_model_name} KM",
        ).opts(
            color=_color,
            line_dash="dashed",
            line_width=5,
            interpolation="steps-post",
        )

        _plots.append(_weibull*_km)


    _plot = reduce(lambda _x, _y: _x + _y, _plots)

    #_plot.opts(
    #    frame_width=600,
    #    frame_height=500,
    #    legend_position="right",
    #    ylabel="Survival probability",
    #    xlabel="Hours",
    #)
    _plot.opts(shared_axes=False)
    return np, reduce


@app.cell(hide_code=True)
def _(df, hv, lifelines, np, pl, reduce, units_c):
    _loglog_plots = []

    _colors = hv.Cycle().values

    _models = (
        df.group_by("model")
          .agg(cnt=pl.len())
          .sort("cnt")
          .top_k(by="cnt", k=10)
          .collect()["model"]
    )


    for _i, _model_name in enumerate(_models):

        _color = _colors[_i % len(_colors)]

        _grp = units_c.filter(
            (pl.col("model") == _model_name)
            & pl.col("exit_hours").is_not_nan() &
            (pl.col('exit_hours') > pl.col('entry_hours'))
        )

        _d = _grp["exit_hours"].to_numpy() + 1
        _e = _grp["event"].to_numpy()
        _t0 = _grp["entry_hours"].to_numpy()

        # Kaplan-Meier estimate
        _kmf = lifelines.KaplanMeierFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
        )

        _t = _kmf.survival_function_.index.to_numpy()
        _s = _kmf.survival_function_.iloc[:, 0].to_numpy()

        # Need:
        #   t > 0
        #   0 < S(t) < 1
        # for both logs to be finite
        _valid = (
            (_t > 0)
            & (_s > 0)
            & (_s < 1)
        )

        _log_t = np.log(_t[_valid])
        _log_neglog_s = np.log(-np.log(_s[_valid]))

        _curve = hv.Curve(
            (_log_t, _log_neglog_s),
            label=_model_name,
        ).opts(
            color=_color,
            line_width=3,
        )

        _loglog_plots.append(_curve)


    _loglog_plot = reduce(
        lambda _x, _y: _x + _y,
        _loglog_plots,
    )

    #_loglog_plot.opts(
    #    frame_width=600,
    #    frame_height=500,
    #    legend_position="right",
    #    xlabel="log(t)",
    #    ylabel="log(-log(S(t)))",
    #)
    _loglog_plot.opts(shared_axes=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding capacity size
    """)
    return


app._unparsable_cell(
    r"""
    units_c_extra = units.join(
        other=df.select(['serial_number', 'capacity_bytes', 'datacenter', 'vault_id']).group_by('serial_number').agg(
            capacity_bytes=pl.first('capacity_bytes'),
            datacenter=p
    """,
    column=None, disabled=False, hide_code=True, name="_"
)


@app.cell(hide_code=True)
def _(df, lifelines, pl, units_c_extra):
    def _():
        _results = []
        _models = df.group_by('capacity_bytes').agg(cnt=pl.len()).sort('cnt').top_k(by='cnt', k=10).collect()['capacity_bytes']
        for model_name, grp in units_c_extra.filter(pl.col('exit_hours').is_not_nan()).group_by('capacity_bytes'):
            if model_name[0] not in _models:
                continue
            d = (grp["exit_hours"].to_numpy() + 1)/24.0
            e = (grp["event"].to_numpy())
            t0 = (grp["entry_hours"].to_numpy())/24.0
    
            wf = lifelines.WeibullFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)
            lf = lifelines.LogNormalFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)
            llf = lifelines.LogLogisticFitter().fit(durations=d, event_observed=e, entry=t0, label=model_name)
    
    
            _results.append({
                "model": model_name[0],
                "n": grp.height, "events": int(e.sum()),
                "weibull_rho": wf.rho_, "weibull_rho_lo": wf.confidence_interval_.iloc[0,0],
                "weibull_rho_hi": wf.confidence_interval_.iloc[0,1],
                "weibull_lambda": wf.lambda_,
                "weibull_aic": wf.AIC_,
                "lognormal_aic": lf.AIC_,
                "loglogistic_aic": lf.AIC_
            })
    
        beta_table_cap = pl.DataFrame(_results).sort("model")
        return beta_table_cap

    beta_table_cap = _()
    return (beta_table_cap,)


@app.cell
def _(beta_table_cap, pl):
    beta_table_cap.with_columns(
        ratio = 100.0*pl.col('events')/pl.col('n')
    )
    return


@app.cell(hide_code=True)
def _(beta_table_cap, df, hv, lifelines, np, pl, reduce, units_c_extra):
    _times = np.linspace(0.0, 2500.0, 1000)

    by_what = 'capacity_bytes'

    _models = (
        df.group_by(by_what)
          .agg(cnt=pl.len())
          .sort("cnt")
          .top_k(by="cnt", k=10)
          .collect()[by_what]
    )

    _colors = hv.Cycle().values

    _ci_plots = []
    _curve_plots = []

    for _i, _model_name in enumerate(_models):

        _color = _colors[_i % len(_colors)]

        # ------------------------------------------------------------------
        # Weibull fitted curve
        # ------------------------------------------------------------------
        _row = beta_table_cap.filter(pl.col('model') == _model_name)

        _lambda = _row["weibull_lambda"].item()
        _rho = _row["weibull_rho"].item()

        _weibull = hv.Curve(
            (
                _times,
                np.exp(-(_times / _lambda) ** _rho),
            ),
            label=f"{_model_name} Weibull",
        ).opts(
            color=_color,
            line_width=5,
        )

        # ------------------------------------------------------------------
        # Kaplan-Meier curve
        # ------------------------------------------------------------------
        _grp = units_c_extra.filter(
            (pl.col(by_what) == _model_name)
            & pl.col("exit_hours").is_not_nan()
            & (pl.col("exit_hours") > pl.col("entry_hours"))
        )

        _d = (_grp["exit_hours"].to_numpy() + 1) / 24.0
        _e = _grp["event"].to_numpy()
        _t0 = _grp["entry_hours"].to_numpy() / 24.0

        _kmf = lifelines.KaplanMeierFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
            alpha=0.05,  # 95% confidence interval
        )

        # KM survival estimate
        _km_times = _kmf.survival_function_.index.to_numpy()
        _km_survival = _kmf.survival_function_.iloc[:, 0].to_numpy()

        _km = hv.Curve(
            (
                _km_times,
                _km_survival,
            ),
            label=f"{_model_name} KM",
        ).opts(
            color=_color,
            line_dash="dashed",
            line_width=5,
        )

        # ------------------------------------------------------------------
        # Kaplan-Meier 95% confidence interval
        # ------------------------------------------------------------------
        _ci = _kmf.confidence_interval_

        _ci_lower = _ci.iloc[:, 0].to_numpy()
        _ci_upper = _ci.iloc[:, 1].to_numpy()

        _km_ci = hv.Area(
            (
                _ci.index.to_numpy(),
                _ci_lower,
                _ci_upper,
            ),
            kdims=["Time"],
            vdims=["Lower", "Upper"],
            label=f"{_model_name} KM 95% CI",
        ).opts(
            color=_color,
            fill_alpha=0.15,
            line_alpha=0,
            show_legend=False,
        )

        # Keep all confidence bands underneath all curves
        _ci_plots.append(_km_ci)
        _curve_plots.extend([_weibull, _km])


    # Put confidence intervals behind the actual curves
    _plots = _ci_plots + _curve_plots

    _plot = reduce(lambda _x, _y: _x * _y, _plots)

    _plot.opts(
        frame_width=600,
        frame_height=500,
        legend_position="right",
        ylabel="Survival probability",
        xlabel="Hours",
    )
    return


@app.cell(hide_code=True)
def _(df, hv, lifelines, np, pl, reduce, units_c_extra):
    _loglog_plots = []

    _colors = hv.Cycle().values

    _models = (
        df.group_by("capacity_bytes")
          .agg(cnt=pl.len())
          .sort("cnt")
          .top_k(by="cnt", k=10)
          .collect()["capacity_bytes"]
    )


    for _i, _model_name in enumerate(_models):

        _color = _colors[_i % len(_colors)]

        _grp = units_c_extra.filter(
            (pl.col("capacity_bytes") == _model_name)
            & pl.col("exit_hours").is_not_nan() &
            (pl.col('exit_hours') > pl.col('entry_hours'))
        ).with_columns()

        _d = _grp["exit_hours"].to_numpy() + 1
        _e = _grp["event"].to_numpy()
        _t0 = _grp["entry_hours"].to_numpy()

        # Kaplan-Meier estimate
        _kmf = lifelines.KaplanMeierFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
        )

        _t = _kmf.survival_function_.index.to_numpy()
        _s = _kmf.survival_function_.iloc[:, 0].to_numpy()

        # Need:
        #   t > 0
        #   0 < S(t) < 1
        # for both logs to be finite
        _valid = (
            (_t > 0)
            & (_s > 0)
            & (_s < 1)
        )

        _log_t = np.log(_t[_valid])
        _log_neglog_s = np.log(-np.log(_s[_valid]))

        _curve = hv.Curve(
            (_log_t, _log_neglog_s),
            label=str(_model_name),
        ).opts(
            color=_color,
            line_width=3,
        )

        _loglog_plots.append(_curve)


    _loglog_plot = reduce(
        lambda _x, _y: _x + _y,
        _loglog_plots,
    )

    #_loglog_plot.opts(
    #    frame_width=600,
    #    frame_height=500,
    #    legend_position="right",
    #    xlabel="log(t)",
    #    ylabel="log(-log(S(t)))",
    #)
    _loglog_plot.opts(shared_axes=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Adding data center, generalized gamma
    """)
    return


@app.cell(hide_code=True)
def _(df, lifelines, np, pl, units_c_extra):
    from scipy.special import gammainc
    from scipy.stats import norm

    def gg_survival(t, mu, ln_sigma, lam):
        sigma = np.exp(ln_sigma)
        z = (np.log(t) - mu) / sigma
        ell = 1.0 / lam**2
        u = ell * np.exp(lam * z)

        if lam > 0:
            return 1 - gammainc(ell, u)
        elif lam < 0:
            return gammainc(ell, u)
        else:
            return 1 - norm.cdf(z)


    def _():
        _results = []

        _models = (
            df.group_by("model")
            .agg(cnt=pl.len())
            .sort("cnt")
            .top_k(by="cnt", k=25)
            .collect()["model"]
        )

        for model_name, grp in units_c_extra.filter(
            pl.col("exit_hours").is_not_nan()
        ).group_by("model"):

            if model_name[0] not in _models:
                continue

            d = grp["exit_hours"].to_numpy() + 1
            e = grp["event"].to_numpy()
            t0 = grp["entry_hours"].to_numpy()

            wf = lifelines.GeneralizedGammaFitter(alpha=0.04).fit(
                durations=d,
                event_observed=e,
                entry=t0,
                label=model_name,
            )

            s = wf.summary

            _results.append({
                "model": model_name[0],
                "n": grp.height,
                "events": int(e.sum()),
                "censored": int((~e.astype(bool)).sum()),

                # mu
                "gamma_mu": wf.mu_,
                "gamma_mu_se": s.loc["mu_", "se(coef)"],
                "gamma_mu_lo": s.loc["mu_", "coef lower 95%"],
                "gamma_mu_hi": s.loc["mu_", "coef upper 95%"],

                # ln(sigma)
                "gamma_ln_sigma": wf.ln_sigma_,
                "gamma_ln_sigma_se": s.loc["ln_sigma_", "se(coef)"],
                "gamma_ln_sigma_lo": s.loc["ln_sigma_", "coef lower 95%"],
                "gamma_ln_sigma_hi": s.loc["ln_sigma_", "coef upper 95%"],

                # sigma, transformed back from log scale
                "gamma_sigma": np.exp(wf.ln_sigma_),
                "gamma_sigma_lo": np.exp(
                    s.loc["ln_sigma_", "coef lower 95%"]
                ),
                "gamma_sigma_hi": np.exp(
                    s.loc["ln_sigma_", "coef upper 95%"]
                ),

                # lambda
                "gamma_lambda": wf.lambda_,
                "gamma_lambda_se": s.loc["lambda_", "se(coef)"],
                "gamma_lambda_lo": s.loc["lambda_", "coef lower 95%"],
                "gamma_lambda_hi": s.loc["lambda_", "coef upper 95%"],

                # fit statistics
                "log_likelihood": wf.log_likelihood_,
                "aic": wf.AIC_,
                "bic": wf.BIC_,
                "median_survival": wf.median_survival_time_,
            })

        return pl.DataFrame(_results).sort("model")


    beta_table_dc = _()
    return beta_table_dc, gg_survival


@app.cell
def _(beta_table_dc):
    beta_table_dc
    return


@app.cell(hide_code=True)
def _(
    beta_table_dc,
    df,
    gg_survival,
    hv,
    lifelines,
    np,
    pl,
    reduce,
    units_c_extra,
):
    # final x value: max observed time
    _by_what = 'model'
    _models = (
        df.group_by(_by_what)
          .agg(cnt=pl.len())
          .sort("cnt")
          .top_k(by="cnt", k=10)
          .collect()[_by_what]
    )
    _max_time = (
        units_c_extra
        .filter(pl.col("model").is_in(_models))
        ["exit_hours"]
        .max()
    )

    _times = np.linspace(1.0, _max_time, 3000)

    _by_what = "model"


    _colors = hv.Cycle().values
    _plots = []

    for _i, _model_name in enumerate(_models):

        if _model_name is None:
            continue

        _color = _colors[_i % len(_colors)]

        # ---------------------------------------------------------
        # Generalized Gamma
        # ---------------------------------------------------------

        _row = beta_table_dc.filter(
            pl.col("model") == _model_name
        )

        _lambda = _row["gamma_lambda"].item()
        _ln_sigma = _row["gamma_ln_sigma"].item()
        _mu = _row["gamma_mu"].item()

        _gg_y = gg_survival(
            _times,
            _mu,
            _ln_sigma,
            _lambda,
        )

        _gg = hv.Curve(
            (_times, _gg_y),
            label=f"{_model_name}",
        ).opts(
            color=_color,
            line_width=4,
        )

        # ---------------------------------------------------------
        # Fit GG again to get survival CI
        # ---------------------------------------------------------

        _grp = units_c_extra.filter(
            (pl.col(_by_what) == _model_name)
            & pl.col("exit_hours").is_not_nan()
            & (pl.col("exit_hours") > pl.col("entry_hours"))
        )

        _d = _grp["exit_hours"].to_numpy() + 1
        _e = _grp["event"].to_numpy()
        _t0 = _grp["entry_hours"].to_numpy()

        _ggf = lifelines.GeneralizedGammaFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
            timeline=_times,
        )

        _gg_ci = _ggf.confidence_interval_survival_function_

        _gg_band = hv.Area(
            (
                _times,
                _gg_ci.iloc[:, 0].to_numpy(),
                _gg_ci.iloc[:, 1].to_numpy(),
            ),
            vdims=["lower", "upper"],
        ).opts(
            color=_color,
            alpha=0.15,
            line_alpha=0,
            width=250,
            height=250
        )

        # ---------------------------------------------------------
        # Kaplan-Meier
        # ---------------------------------------------------------

        _kmf = lifelines.KaplanMeierFitter().fit(
            durations=_d,
            event_observed=_e,
            entry=_t0,
        )

        _km_times = _kmf.survival_function_.index.to_numpy()
        _km_y = _kmf.survival_function_.iloc[:, 0].to_numpy()

        _km = hv.Curve(
            (_km_times, _km_y),
        ).opts(
            color="gray",
            line_dash="dashed",
            line_width=3,
            interpolation="steps-post",
        )

        # KM confidence interval
        _km_ci = _kmf.confidence_interval_

        _km_band = hv.Area(
            (
                _km_ci.index.to_numpy(),
                _km_ci.iloc[:, 0].to_numpy(),
                _km_ci.iloc[:, 1].to_numpy(),
            ),
            vdims=["lower", "upper"],
        ).opts(
            color="gray",
            alpha=0.12,
            line_alpha=0,
            xrotation=30,
            xlabel="",
            ylabel="",

        )

        _plots.append(
            _gg_band
            * _km_band
            * _gg
            * _km
        )


    _plot = reduce(lambda _x, _y: _x + _y, _plots)

    plots_aft = _plot.opts(
        shared_axes=False,
    ).cols(5)
    return (plots_aft,)


@app.cell
def _(plots_aft):
    from holoviews import save
    save(plots_aft, './backblaze_drive_stats/results/estimators/aft_gg_models.html')
    plots_aft
    return


@app.cell
def _(mo, np):
    times_temp = np.linspace(0.0, 10.0, 100)
    lambda_sel = mo.ui.slider(start=1e-3, stop=1, step=0.01, full_width=True)
    rho_sel = mo.ui.slider(start=1e-3, stop=100, step=1, full_width=True)

    mo.vstack([lambda_sel, rho_sel])
    return lambda_sel, rho_sel, times_temp


@app.cell
def _(hv, lambda_sel, np, rho_sel, times_temp):
    hv.Curve(
        (times_temp, 
        np.exp(-(times_temp / lambda_sel.value) ** rho_sel.value)
        ),
    ).opts(width=500, height=500,
        xlim=(0, 10), ylim=(0, 1)
          )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
