import marimo

__generated_with = "0.23.15"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo
    import arviz as az
    import pymc as pm
    import glob 
    import polars as pl
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")
    GAP_TOLERANCE = 7    
    MAX_BRIDGED   = 5    
    MAX_SPELLS    = 10    
    return GAP_TOLERANCE, MAX_BRIDGED, MAX_SPELLS, glob, np, pl, pm


@app.cell(hide_code=True)
def _(glob, pl):
    lfs = [
        pl.scan_parquet(f, hive_partitioning=True)
        for f in glob.glob(
            "backblaze_drive_stats/*2025/*.parquet",
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
    return (spells,)


@app.cell
def _(pl, spells):
    spells_c = spells.with_columns(
        manufacturer=pl.when(
            pl.col('model').str.starts_with('ST')
        ).then(pl.lit('ST')).otherwise(
            pl.col("model").str.split(" ").list.first()
        )
    ).filter(
        pl.col('exit_hours').is_not_nan()
    ).collect()
    return (spells_c,)


@app.cell
def _(np, pm, spells_c):
    model_to_mfr = (
        spells_c.select(['model', 'manufacturer'])
        .unique()
        .sort('model')
    )

    manufacturers_unique_values, manufacturers_idx_spells = np.unique(spells_c['manufacturer'], return_inverse=True)
    models_unique_values, models_idx_spells = np.unique(spells_c['model'], return_inverse=True)
    manufacturers_unique_, mfr_idx_arr = np.unique(model_to_mfr['manufacturer'], return_inverse=True)

    coords = {
        'spells': np.arange(spells_c.height),
        'model': models_unique_values,
        'manufacturer': manufacturers_unique_values
    }

    def f_w(t_, lambda_m, k_m):
        z = t_ / lambda_m
        return (k_m / lambda_m) * z**(k_m - 1) * pm.math.exp(-z**k_m)


    def S_w(t_, lambda_m, k_m):
        return pm.math.exp(-(t_/lambda_m)**k_m)

    with pm.Model(coords=coords) as model1:
        event_data = pm.Data(name='event_data', dims='spells', value=spells_c['event'].to_numpy())
        exit_h_data = pm.Data(name='exit_data', dims='spells', value=spells_c['exit_hours'].to_numpy())
        entry_h_data = pm.Data(name='entry_data', dims='spells', value=spells_c['entry_hours'].to_numpy())
        mfr_idx = pm.Data('mfr_idx_data', mfr_idx_arr, dims='model')

        tau = pm.HalfNormal(name='tau', sigma=0.5)
        u_m = pm.Normal(name='u_m', dims='model', mu=0.0, sigma=1.0)
        v_m = pm.Normal(name='v_m', dims='model', mu=0.0, sigma=1.0)
        mu_j = pm.Normal(name='mu_j', mu=8.0, sigma=1.0, dims='manufacturer')
        nu = pm.Normal(name='nu', mu=1.0, sigma=1.0)
        rho = pm.HalfNormal(name='rho', sigma=1.0)

        a_m = pm.Deterministic(
            name='a_m', dims='model', var=(
                mu_j[mfr_idx] + tau*u_m
            )
        )
        b_m = pm.Deterministic(
            name='b_m', dims='model', var=(
                nu + rho*v_m
            )
        )
        lambda_m = pm.Deterministic(name='lambda_m', var=pm.math.exp(a_m), dims='model')
        k_m = pm.Deterministic(name='k_m', var=pm.math.exp(b_m), dims='model')
    
        loglik = (
            event_data*pm.math.log(
                f_w(exit_h_data, lambda_m, k_m)
            ) + 
            (1.0 - event_data)*pm.math.log(
                S_w(exit_h_data, lambda_m, k_m)
            ) -
            pm.math.log(
                S_w(entry_h_data, lambda_m, k_m)
            )
        )
    
    return (models_unique_values,)


@app.cell
def _(spells_c):
    spells_c
    return


@app.cell
def _(v_):
    v_
    return


@app.cell
def _(idx_):
    idx_
    return


@app.cell
def _():
    #model_to_mfr = (
    #    spells_c.select(['model', 'manufacturer'])
    #    .unique()
    #    .sort('model')
    #)
    #np.unique(model_to_mfr['manufacturer'], return_inverse=True)
    return


@app.cell
def _(models_unique_values):
    models_unique_values
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
