import os
import warnings
import datetime
import pytz
import re
import pickle
import functools
from collections import OrderedDict

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
import numpy as np
import pandas as pd
import numpy.ma as ma
import polars as pl

import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import dask.array as da

import pyproj
import re

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors

from tqdm import tqdm

## https://hdfeos.org/software/pyhdf.php
warnings.simplefilter("ignore")


def get_dates_pl(dates):
    dates_pl = (
        pl.from_pandas(
            pd.DataFrame(
                {
                    "date": pd.to_datetime(dates),
                    "year": pd.to_datetime(dates).year,
                    "month": pd.to_datetime(dates).month,
                }
            )
        )
        .with_columns(
            month=pl.col("month").cast(str).str.zfill(2),
            day=pl.lit("01"),
        )
        .with_columns(
            year_month=pl.concat_str(
                [pl.col("year"), pl.col("month"), pl.col("day")], separator="-"
            ).str.to_date("%Y-%m-%d"),
        )
        .sort(["date"])
    )

    year_months = np.sort(dates_pl["year_month"].unique().to_numpy())
    return dates_pl, year_months


def process_data(data, n_blocks_per_row):
    """
    Y: ndvi
    X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane
    shape = (n_blocks_per_row, n_blocks_per_row, n_timeperiods, n_vars, n_changepoints)
    """
    mu = data.mean(axis=2, keepdims=True)
    mu
    sigma = data.std(axis=2, keepdims=True)
    sigma
    X_std = (data - mu) / sigma

    if X_std.shape[0] == 1:
        # broadcast out values for the satellite grid
        X_std = X_std * np.ones(shape=(n_blocks_per_row, n_blocks_per_row, 1, 1, 1))

    changepoints = (
        pd.Series(X_std.flatten())
        .describe()[["25%", "50%", "75%"]]
        .values[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]
    )

    I_data = (X_std > changepoints).astype(np.int16)

    print(f"changepoints.shape: {changepoints.shape}")
    print(f"I_data.shape: {I_data.shape}")
    print(f"X_std.shape: {X_std.shape}")

    print(f"changepoints: {changepoints.squeeze()}")

    return changepoints, I_data, X_std


def get_fourier_components(
    n_timeperiods,
    n_blocks_per_row,
    periodicity,
    n_components,
):
    """
    time_base: linspace between 0 and 1 with len = timeperiods
    periodicity: how often the seasonality repeats in timeperiods
    n_components: this dictates how many sine and cosine waves are weighted
        together to get the curve, move components = more flexible. Rules of
        thumb are 3 for day of the week (7 period repeats) and 10 for time
        of year (365.25 period repeats). There ends up being n_components * 2
        columns, a sine and cosine wave for each component.

    returns shape (n_timeperiods, n_components*2)
    """

    """
Y: ndvi
X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane
shape = (n_blocks_per_row, n_blocks_per_row, n_timeperiods, n_vars, n_changepoints)
"""
    timebase = np.linspace(start=0, stop=1, num=n_timeperiods)[:, np.newaxis]
    n_parameter = np.arange(n_components)[np.newaxis, :] + 1
    p_parameter = periodicity / len(timebase)
    internals = (2 * np.pi * n_parameter * timebase) / p_parameter
    fourier_components = np.concatenate(
        [np.cos(internals), np.sin(internals)],
        axis=1,
    )
    fourier_components = np.repeat(
        a=np.repeat(
            a=fourier_components[np.newaxis, np.newaxis, ...],
            repeats=n_blocks_per_row,
            axis=0,
        ),
        repeats=n_blocks_per_row,
        axis=1,
    )[..., np.newaxis]
    fourier_components = np.swapaxes(a=fourier_components, axis1=3, axis2=4)
    fourier_components = tf.convert_to_tensor(
        value=fourier_components,
        dtype=tf.float32,
    )
    return fourier_components


def plot_contour(
    cities,
    data_base,
    data_learned,
    lat,
    lon,
    start_lat,
    stop_lat,
    start_lon,
    stop_lon,
    data=None,
    title=None,
    levels=None,
    y=0.67,
):
    if data is None:
        data = np.zeros_like(data_base)
    data[
        start_lon:stop_lon,
        start_lat:stop_lat,
    ] = data_learned

    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(subplot_kw=dict(projection=proj), figsize=(10, 20))
    ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())

    for i in range(cities.shape[0]):
        lon_city = cities[i, 3]
        lat_city = cities[i, 2]
        if (
            (lon_city > lon.min())
            and (lon_city < lon.max())
            and (lat_city > lat.min())
            and (lat_city < lat.max())
        ):
            ax.plot(
                lon_city,
                lat_city,
                "bo",
                markersize=6,
                color="red",
                transform=ccrs.Geodetic(),
            )
            ax.text(x=cities[i, 3], y=cities[i, 2], s=cities[i, 0])

    if levels is not None:
        plt.contourf(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="YlGn",
            levels=levels,
        )
    else:
        plt.contourf(
            lon,
            lat,
            data,
            transform=ccrs.PlateCarree(),
            cmap="YlGn",
        )

    ax.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=4)
    ax.add_feature(cfeature.OCEAN)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1)
    ax.add_feature(cartopy.feature.RIVERS, edgecolor="blue", linewidth=0.5)

    states_provinces = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(
        states_provinces,
        edgecolor="black",
        zorder=10,
        linestyle="-",
        linewidth=2,
        alpha=0.5,
    )

    ax.gridlines(draw_labels=True)
    plt.colorbar(ax=ax, shrink=0.25, pad=0.07, label='Probability',)
    plt.suptitle(title, y=y, fontsize=28)


def get_joint_distribution(
    X_fourier,
    X_data,
    changepoints,
    I_data,
    n_blocks_per_row,
    n_vars,
):
    """
    Y: ndvi
    X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane
    shape = (n_blocks_per_row, n_blocks_per_row, n_timeperiods, n_vars, n_changepoints)
    """
    scale_prior_fourier = 5
    scale_prior_piecewise = 5
    scale_prior_error = 10

    n_changepoints = changepoints.shape[4]

    y_dist = functools.partial(
        y_distribution,
        # data
        X_fourier=X_fourier,
        X_data=X_data,
        changepoints=changepoints,
        I_data=I_data,
    )

    model = OrderedDict(
        # fourier curves
        fourier_weights_prior=tfd.HalfNormal(
            scale=scale_prior_fourier,
            name="fourier_weights_prior",
        ),
        b_fourier=lambda fourier_weights_prior: tfd.MultivariateNormalDiag(
            loc=tf.zeros(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, 1, X_fourier.shape[4]]
            ),
            scale_diag=fourier_weights_prior
            * tf.ones(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, 1, X_fourier.shape[4]]
            ),
            name="b_fourier",
        ),
        # b0
        b0_prior=tfd.HalfNormal(
            scale=scale_prior_piecewise * tf.ones(shape=[1, 1, 1, n_vars, 1]),
            name="b0_prior",
        ),
        b0=lambda b0_prior: tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, 1]),
            scale_diag=b0_prior
            * tf.ones(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, 1],
            ),
            name="b0",
        ),
        # b1
        b1_prior=tfd.HalfNormal(
            scale=scale_prior_piecewise * tf.ones(shape=[1, 1, 1, n_vars, 1]),
            name="b1_prior",
        ),
        b1=lambda b1_prior: tfd.MultivariateNormalDiag(
            loc=tf.zeros(shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, 1]),
            scale_diag=b1_prior
            * tf.ones(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, 1],
            ),
            name="b1",
        ),
        # b2
        # allow for different priors for each changepoint, the hierarchy is on the
        # blocks, now across the change points, this should create more flexibility
        b2_prior=tfd.HalfNormal(
            scale=scale_prior_piecewise
            * tf.ones(shape=[1, 1, 1, n_vars, n_changepoints]),
            name="b2_prior",
        ),
        b2=lambda b2_prior: tfd.MultivariateNormalDiag(
            loc=tf.zeros(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, n_changepoints]
            ),
            scale_diag=b2_prior
            * tf.ones(
                shape=[n_blocks_per_row, n_blocks_per_row, 1, n_vars, n_changepoints],
            ),
            name="b2",
        ),
        error=tfd.HalfNormal(
            scale=scale_prior_error
            * tf.ones(
                shape=[
                    n_blocks_per_row,
                    n_blocks_per_row,
                    1,
                    1,
                    1,
                ]
            )
        ),
        y=lambda b_fourier, b0, b1, b2, error: y_dist(
            b_fourier=b_fourier,
            b0=b0,
            b1=b1,
            b2=b2,
            error=error,
        ),
    )

    joint_distribution = tfd.JointDistributionNamed(
        model=model,
        use_vectorized_map=True,
        batch_ndims=0,
    )

    return joint_distribution


def get_piecewise_contribution(b0, b1, b2, I_data, changepoints, X_data):
    """
    Y: ndvi
    X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane
    shape = (n_blocks_per_row, n_blocks_per_row, n_timeperiods, n_vars, n_changepoints)
    """
    intercept = b0 + tf.reduce_sum(
        input_tensor=(b2 * I_data * (-1 * changepoints)),
        axis=4,
        keepdims=True,
    )
    slope = b1 + tf.reduce_sum(
        input_tensor=(b2 * I_data),
        axis=4,
        keepdims=True,
    )
    contribution = tf.reduce_sum(
        input_tensor=intercept + (slope * X_data),
        axis=3,
        keepdims=True,
    )
    return contribution


def y_distribution(
    # data
    X_fourier,
    X_data,
    changepoints,
    I_data,  # indicator matrix for where you are in relation to changepoints
    # parameters
    b_fourier,
    b0,
    b1,
    b2,
    # scale
    error,
):
    """
    Y: ndvi, pm25, etc
    X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane
    shape = (n_blocks_per_row, n_blocks_per_row, n_timeperiods, n_vars, n_changepoints)

    Formula to get ndvi

    changepoints_trend, changepoints_propane, changepoints_clouds, changepoints_rain, changepoints_heat,
    I_trend, I_propane, I_clouds, I_rain, I_heat,
    X_trend, X_propane, X_clouds, X_rain, X_heat

    X_trend: np.linspace(start=0, stop=1, num=n_timeperiods)
    """
    # Fourier Curves -----------------------------------------------------------
    fourier = tf.reduce_sum(
        input_tensor=b_fourier * X_fourier,
        axis=4,
        keepdims=True,
    )

    # Piecewise Curves -----------------------------------------------------------
    piecewise = get_piecewise_contribution(
        b0=b0,
        b1=b1,
        b2=b2,
        I_data=I_data,
        changepoints=changepoints,
        X_data=X_data,
    )

    # Y:  ------------------------------------------------------------------
    y_mean = fourier + piecewise

    y = tfd.MultivariateNormalDiag(
        loc=y_mean,
        scale_diag=error,
        name="y",
    )
    return y


def get_piecewise_posterior(
    b0,
    b1,
    b2,
    I_data,
    changepoints,
    X_data,
    chunks,
):
    """
    Y: ndvi
    X: fourier components, piecewise_vars = trend, clouds, rain, heat, propane

    n_blocks short for n_blocks_per_row
    shape = (n_samples, n_blocks, n_blocks, n_timeperiods, n_vars, n_changepoints)

    """
    # np.newaxis for the samples
    I_data = da.from_array(I_data[np.newaxis, ...], chunks=chunks)
    X_data = da.from_array(X_data[np.newaxis, ...], chunks=chunks)
    changepoints = da.from_array(changepoints[np.newaxis, ...], chunks=chunks)

    intercept = b0 + (b2 * I_data * (-1 * changepoints)).sum(
        axis=5,
        keepdims=True,
    )

    slope = b1 + (b2 * I_data).sum(
        axis=5,
        keepdims=True,
    )

    contribution = intercept + (slope * X_data)
    return contribution, intercept, slope


@tf.function(autograph=False, experimental_compile=False)
def run_chain(
    init_state,
    step_size,
    target_log_prob_fn,
    unconstraining_bijectors,
    num_steps=500,
    burnin=50,
):

    def trace_fn(_, pkr):
        return (
            pkr.inner_results.inner_results.target_log_prob,
            pkr.inner_results.inner_results.leapfrogs_taken,
            pkr.inner_results.inner_results.has_divergence,
            pkr.inner_results.inner_results.energy,
            pkr.inner_results.inner_results.log_accept_ratio,
        )

    kernel = tfp.mcmc.TransformedTransitionKernel(
        inner_kernel=tfp.mcmc.NoUTurnSampler(target_log_prob_fn, step_size=step_size),
        bijector=unconstraining_bijectors,
    )

    hmc = tfp.mcmc.DualAveragingStepSizeAdaptation(
        inner_kernel=kernel,
        num_adaptation_steps=burnin,
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            inner_results=pkr.inner_results._replace(step_size=new_step_size)
        ),
        step_size_getter_fn=lambda pkr: pkr.inner_results.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.inner_results.log_accept_ratio,
    )

    # Sampling from the chain.
    chain_state, sampler_stat = tfp.mcmc.sample_chain(
        num_results=num_steps,
        num_burnin_steps=burnin,
        current_state=init_state,
        kernel=hmc,
        trace_fn=trace_fn,
    )
    return chain_state, sampler_stat


def get_post(jd, samples, n_samples, n_chains, chunks):
    parameter_names = [
        name for name in jd.parameters["model"].keys() if "y" not in name
    ]
    post = {
        param: (
            (
                da.from_array(
                    sample.numpy().reshape(
                        n_samples * n_chains,
                        sample.shape[2],
                        sample.shape[3],
                        sample.shape[4],
                        sample.shape[5],
                        sample.shape[6],
                    ),
                    chunks=chunks,
                )
            )
            if param != "fourier_weights_prior"
            else (
                da.from_array(
                    sample.numpy().reshape(
                        n_samples * n_chains,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ),
                    chunks=chunks,
                )
            )
        )
        for param, sample in zip(parameter_names, samples)
    }
    print(f"len(samples): {len(samples)}")
    for param, sample in post.items():
        print(f"{param}.shape: {sample.shape}")

    return post
