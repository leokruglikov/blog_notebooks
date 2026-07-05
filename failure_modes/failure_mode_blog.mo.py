import marimo

__generated_with = "0.23.8"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import marimo as mo
    import polars as pl
    import hvplot.polars
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    import holoviews as hv
    import numpy as np
    import pymc as pm
    import hvplot.xarray
    import arviz as az

    return hv, mo, np, pl, pm, sm, smf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read and prep the data
    """)
    return


@app.cell
def _(pl):
    COUNTS_DF = pl.read_parquet('./failure_modes.parquet')
    COUNTS_DF
    return (COUNTS_DF,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The contains the number of the mode occured at some rank $r$.
    E.g. For example, the number of successes/passes on the first trial is the biggest and around 1.7M. We denote the number of failures of mode $m$ observed at rank $r$ as $C_{m,r}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## First evaluation
    The first evaluation of the decay factor of mode $m$ is given by
    $$
    \hat{q}_m= (\frac{C_{m,K}}{C_{m,1}})^{1/(K-1)}
    $$
    """)
    return


@app.cell
def _(COUNTS_DF, pl):
    COUNTS_DF.group_by('failure_mode').agg(
        ratio = ((pl.col('count').filter(
            pl.col('rank') == pl.col('rank').max()
        ).first())/(
            pl.col('count').filter(
                pl.col('rank') == pl.col('rank').min()
            ).first()
        )).pow(1.0/3.0)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This already gives the first idea of the 'decay rate' $\hat{q}$.
    """)
    return


@app.cell
def _(COUNTS_DF, pl):
    fig_failures = COUNTS_DF.filter(
        pl.col('failure_mode').is_in(['FMODE_D', 'FMODE_B'])
    ).with_columns(
        pl.col('rank').cast(pl.String).alias('rank')
    ).sort('rank').hvplot.bar(x='rank',y='count', by='failure_mode').opts(xrotation=50)
    #hv.save(fig_failures, 'failure_modes_decay.html', fmt='html')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Frequentist model
    Before using Bayes to fit the model, we will use the plain frequentist method for diversity's sake.

    The model under question is
    $$
    \log \mu = \alpha + \beta (r-1)
    $$
    which is exactly the GLM with the Poisson link function.
    """)
    return


@app.cell
def _(COUNTS_DF):
    COUNTS_DF
    return


@app.cell
def _(COUNTS_DF, sm, smf):
    poisson_model = smf.glm(
        formula="count ~ 0 + C(failure_mode) + C(failure_mode):rank0 ",
        data=COUNTS_DF.to_pandas(),
        family=sm.families.Poisson()
    )
    fit_poisson_model = poisson_model.fit()
    return (fit_poisson_model,)


@app.cell
def _(fit_poisson_model):
    fit_poisson_model.summary()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We see the resulting coefficients in the link function space making sense with the underlying data. Indeed, $\alpha$ for the mode SUCCESS is around $14$, which in the event space
    is given by $e^{14}\approx 1.2\text{M}$, which agrees with our data as null intercept.

    All of the $\beta$'s are negative, which agrees with the underlying logic of decaying counts.
    """)
    return


@app.cell
def _(fit_poisson_model):
    fit_poisson_model._results.params
    return


@app.cell(hide_code=True)
def _(COUNTS_DF, fit_poisson_model, hv, np, pl):
    tmp_alpha = fit_poisson_model._results.params[0]
    tmp_beta = fit_poisson_model._results.params[5]
    tmp_xs = np.array([0,1,2,3])

    poisson_empirical_vs_fit_freq_fig = (
        COUNTS_DF.filter(
            pl.col('failure_mode').is_in(['FMODE_A'])
        ).sort('rank').hvplot.bar(x='rank0',y='count').opts(xrotation=50)*
       hv.Curve((
           tmp_xs, 
           np.exp(tmp_alpha)*np.exp(tmp_beta*tmp_xs)
       )).opts(color='black')*
       hv.Scatter((
           tmp_xs, 
           np.exp(tmp_alpha)*np.exp(tmp_beta*tmp_xs)
       )).opts(color='black', size=10)
    )
    #hv.save(poisson_empirical_vs_fit_freq_fig, filename='./res/freq_fit_vs_empirical.html', fmt='html')
    poisson_empirical_vs_fit_freq_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As natural and useful extension to the model fitting, we can for instance add the intervals.
    An important step to do when working with Poisson GLM, is to check
    """)
    return


@app.cell
def _():
    return


@app.cell(column=1, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian approach - Poisson
    We will now model the problem using the Bayesian approach.

    The model in question is of the form
    $$
    C_i \sim \text{Pois}(\mu_i) \\
    \log(\mu_{i}) = \alpha_{m[i]} + \beta_{m[i]}(r-1)
    $$
    so every entry in the table $C_i$ follows as poisson with the particular $\mu_i$ of this row.
    """)
    return


@app.cell(hide_code=True)
def _(COUNTS_DF, np, pm):
    # alpha.shape == n_modes
    # beta.shape == n_modes
    # mode_idx.shape == n_rows
    # r0.shape == n_rows
    # count.shape == n_rows

    # prep the data
    failure_modes, failure_modes_idx = np.unique(COUNTS_DF['failure_mode'], return_inverse=True)

    coords_bayesian_poisson = {
        'failure_modes': failure_modes,
        'obs_id': list(range(len(COUNTS_DF['failure_mode'].to_list())))
    }


    with pm.Model(coords=coords_bayesian_poisson) as pois_model:
        counts_data = pm.Data(name='counts_data', value=COUNTS_DF['count'].to_numpy(), dims='obs_id')
        failure_modes_idx_data = pm.Data(name='mode_idx', value=failure_modes_idx)
        r0_data = pm.Data(name='r0_data', value=COUNTS_DF['rank0'].to_numpy(), dims='obs_id')

        #alpha_pois_priors = pm.Exponential(name='alpha_priors', lam=1.0/10.0, dims='failure_modes')
        beta_pois_priors_raw = pm.Exponential(name='beta_priors_raw', lam=1.0/10.0, dims='failure_modes')
        #alpha = pm.Normal(name='alpha', mu=alpha_pois_priors, sigma=2.0, dims='failure_modes')
        alpha = pm.Exponential(name='alpha', lam=1.0/20.0, dims='failure_modes')
        beta = pm.Deterministic(name='beta', var=-beta_pois_priors_raw, dims='failure_modes')

        logmu = alpha[failure_modes_idx] + beta[failure_modes_idx]*r0_data

        y_obs = pm.Poisson(name='y_obs', mu=np.exp(logmu), observed=counts_data, dims='obs_id')

        idata_pois_prior = pm.sample_prior_predictive()
        idata_pois = pm.sample()
        idata_ppc = pm.sample_posterior_predictive(idata_pois)
    return failure_modes, failure_modes_idx, idata_pois, idata_ppc


@app.cell
def _(idata_pois, mo):
    which_mode_plot='FMODE_B'
    mo.vstack(
        [
            mo.md("""
            ### Some inferred parameters
            """),
            idata_pois.posterior.alpha.sel(failure_modes=which_mode_plot).hvplot.hist(),
        ]
    )
    return


@app.cell(hide_code=True)
def _(hv, idata_pois, mo, np):
    which_mode_plot_1 = "FMODE_A"
    which_mode_plot_2 = "FMODE_C"

    bayesian_q_comparison =    (
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_1)).hvplot.hist(alpha=0.7, color='blue')*
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_2)).hvplot.hist(alpha=0.7, color='red')
        )
    hv.save(bayesian_q_comparison, './res/bayesian_q_comparison.html', fmt='html')
    mo.vstack([
        mo.md(rf"""We can obtain the actual parameters:
        $$
        q_m = \exp(\beta _m)
        $$
        and show for {which_mode_plot_1} and {which_mode_plot_2} (blue and red respectively)
        """),
        (
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_1)).hvplot.hist(alpha=0.7, color='blue')*
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_2)).hvplot.hist(alpha=0.7, color='red')
        )
    ])
    return which_mode_plot_1, which_mode_plot_2


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And we can look at what is the probability that one of the failure modes is higher.
    """)
    return


@app.cell(hide_code=True)
def _(idata_pois, mo, np, which_mode_plot_1, which_mode_plot_2):

    is_more_probable =\
    (
        (
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_1)).stack(sample=('chain','draw')) - 
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_2)).stack(sample=('chain','draw'))
        ) < 0
    ).mean()

    mo.vstack([
        mo.md("""
        The probability that the effect 1 has a greater "persistance" is given by 
        $$
        P(q_2>q_1)
        $$
        which is nothing but the difference of the two effects 
        $$
        q_1-q_2<0
        $$
        and can be visualized with pdf.
        """),
        (
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_1)).stack(sample=('chain','draw')) - 
            np.exp(idata_pois.posterior.beta.sel(failure_modes=which_mode_plot_2)).stack(sample=('chain','draw'))
        ).hvplot.hist(),
        mo.md(f"""
        In this case, the probability of q_2 being greater than q_1 is {is_more_probable.item()}, which, arguably, is unlikely!
        """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Bayesian overdispersion
    We can then check for overdispersion using the Bayesian approach.
    For that, we need to do a posterior predictive check.
    The _"classical"_ definition of the Pearson residuals
    $$
    r_i^P = \frac{y_i - \hat{\mu}_i}{\sqrt{ \hat{\mu}_i }}
    $$
    , which, in the Bayesian framework, is mapped to the overdispersion statistic $T$
    $$
    T_\text{obs}^{(s)} = \sum_i \frac{(y_i - \mu_i^{(s)})^2}{\mu_i^{(s)}}
    $$
    where $\mu_i^{(s)}$ is the sample parameter of the fitted Poisson.
    And from the ppc:
    $$
    T_\text{ppc}^{(s)} = \sum_i \frac{(\tilde{y}_i - \mu_i^{(s)})^2}{\mu_i^{(s)}}
    $$
    where $\tilde{y}_i$ - the ppc-generated sample.
    """)
    return


@app.cell(hide_code=True)
def _(
    COUNTS_DF,
    failure_modes,
    failure_modes_idx,
    hv,
    idata_pois,
    idata_ppc,
    np,
    pl,
):
    def bayesian_overdispersion(idata, idata_ppc, which_mode='FMODE_A'):
        idx_ = ( failure_modes[failure_modes_idx] == which_mode )
        counts_df_mode = COUNTS_DF.filter(pl.col('failure_mode')==which_mode)
        assert len(counts_df_mode) == np.sum(idx_)
        T_obs = 0
        T_ppc = 0

        for i in range(len(idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_])):
            y_i = list(counts_df_mode.iter_rows())[i][2]
            r_1 = list(counts_df_mode.iter_rows())[i][-1]
            mu_i_s = np.exp(
                idata_pois.posterior.alpha.stack(sample=('chain', 'draw')).sel(failure_modes=which_mode)+\
                idata_pois.posterior.beta.stack(sample=('chain', 'draw')).sel(failure_modes=which_mode)*(r_1)
            )
            y_tilde_i = idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][i]
            T_obs_ = (
                ( y_i - mu_i_s )**2 / mu_i_s
            )
            T_ppc_ = (
                ( y_tilde_i - mu_i_s )**2 / mu_i_s
            )
            T_obs += T_obs_
            T_ppc += T_ppc_

        return T_obs, T_ppc


    idx_ = failure_modes[failure_modes_idx] == 'FMODE_A'
    idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][0]
    T_overdisp_obs, T_overdisp_ppc = bayesian_overdispersion(idata_pois, idata_ppc)

    bayesian_overdispersion_plot = pl.DataFrame(
        {
            'obs': T_overdisp_obs.to_numpy(),
            'ppc': T_overdisp_ppc.to_numpy(),
            'frac': T_overdisp_obs.to_numpy()/T_overdisp_ppc.to_numpy()
        }
    ).hvplot.hist(y='frac', bins=400 ).opts(title='Overdispersion')
    hv.save(bayesian_overdispersion_plot, filename='./res/bayesian_overdispersion.html', fmt='html')
    bayesian_overdispersion_plot
    return (idx_,)


@app.cell
def _(idata_ppc, idx_):
    idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][0].hvplot.hist(bins=40)*\
    idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][1].hvplot.hist(bins=40)*\
    idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][2].hvplot.hist(bins=40)*\
    idata_ppc.posterior_predictive.y_obs.stack(sample=('chain', 'draw'))[idx_][3].hvplot.hist(bins=40)
    return


@app.cell
def _():
    return


@app.cell(column=2, hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bayesian approach - Geometric/Bernoulli
    We now work with a more complete data set instead of the aggregated one
    """)
    return


@app.cell
def _(pl):
    COUNTS_DF_FULL = pl.read_parquet('./failure_modes_full.parquet')
    COUNTS_DF_FULL = COUNTS_DF_FULL.with_columns(
        outcome_flag=(
            pl.col('FMODE')=="SUCCESS"
        ).cast(pl.Int8),
    )

    COUNTS_DF_FULL = COUNTS_DF_FULL.sort('TIMESTAMP').with_columns(
        attempt=pl.cum_count("ID").over("ID")
    )

    COUNTS_DF_FULL = COUNTS_DF_FULL.sort('TIMESTAMP').with_columns(
        si=(
            (pl.col("FMODE") == "SUCCESS")
            .cast(pl.Int8)
            .max()
            .over("ID")
        ),
        fi=(
            (pl.col('FMODE')!="SUCCESS")
            .cast(pl.Int8)
            .sum()
            .over("ID")
        ),
        m=(
          (pl.col("FMODE")).mode().first().over("ID")  
        )
    )

    COUNTS_DF_FULL = COUNTS_DF_FULL.unique("ID")
    return (COUNTS_DF_FULL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The likelihood entering in the bayesian model is given by
    $$
    \ell_i = s_i \log(p_{m_i}) + f_i \log(1-p_{m_i})
    $$
    where $p_{m_i} \sim \text{logit}(\theta_m)$ some prior.
    """)
    return


@app.cell
def _(COUNTS_DF_FULL, np, pm):
    failure_modes_full, failure_modes_idx_full = np.unique(COUNTS_DF_FULL['FMODE'], return_inverse=True)
    coords_bayesian_geom = {
        "obs_id": list(range(len(COUNTS_DF_FULL))) ,
        "failure_mode": failure_modes_full
    }

    with pm.Model(coords=coords_bayesian_geom) as geom_model:
        si_data = pm.Data(name='si_data', value=COUNTS_DF_FULL['si'].to_numpy(), dims="obs_id")
        fi_data = pm.Data(name='fi_data', value=COUNTS_DF_FULL['fi'].to_numpy(), dims="obs_id")
        failure_modes_idx_data_full = pm.Data(name='mode_idx', value=failure_modes_idx_full, dims='obs_id')

        theta = pm.Normal(name='theta', mu=0.0, sigma=1.0, dims='failure_mode')
        p = pm.Deterministic( name='p', var=pm.math.sigmoid(theta), dims='failure_mode' )

        p_i = p[failure_modes_idx_data_full]

        loglik_i = (
            si_data*pm.math.log(p_i) + fi_data*pm.math.log(1.0 - p_i)
        )
        pm.Potential(name='loglik', var=loglik_i.sum())

        idata_geom = pm.sample(
        
        )
    return failure_modes_full, idata_geom


@app.cell
def _(failure_modes_full, idata_geom, idata_pois, np):
    from matplotlib.pyplot import xlim
    from functools import reduce
    plots_ = []
    plots_geom_only = []
    for fmode_ in failure_modes_full:
        geom = (
            (1 - idata_geom.posterior.p.sel(failure_mode=fmode_))
            .rename("Geom")
            .hvplot.hist()
        )

        pois = (
            np.exp(idata_pois.posterior.beta.sel(failure_modes=fmode_))
            .rename("Poiss")
            .hvplot.hist()
        )

        plot_ = (
            geom * pois
        ).opts(
            xlim=(0.0, 1.0),
            frame_height=150,
            show_grid=True,
            title=str(fmode_),
        )

        plots_.append(plot_)

    pois_geom_plots = reduce(lambda x, y: x + y, plots_).cols(1)
    geom_only_plot = (
            (1 - idata_geom.posterior.p.sel(failure_mode='FMODE_B'))
            .rename("Geom")
            .hvplot.hist()
        ).opts(
            xlim=(0.0, 1.0),
            frame_height=150,
            show_grid=True,
        )
    #hv.save(pois_geom_plots, filename="res/pois_geom_comparison.html", fmt="html")
    #hv.save(geom_only_plot, filename="res/geom_comparison.html", fmt="html")
    pois_geom_plots
    return


@app.cell
def _(COUNTS_DF_FULL, pl):
    COUNTS_DF_FULL.filter(
        pl.col('m') == 'SUCCESS'
    )
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
