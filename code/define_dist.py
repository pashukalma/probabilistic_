import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax.scipy.stats import beta
from scipy.stats import norm

import numpy as np
from scipy.stats import norm, laplace, t

import pymc as pm
import arviz as az

import matplotlib.pyplot as plt
import seaborn as sns

from ipywidgets import interact

"""Beta Distribution"""

import math
def graph(data):
    x = data['x']
    a_list = data['a_list']
    b_list = data['b_list']
    props = data['props']

    num_plots = len(a_list)
    ncols = 3
    nrows = math.ceil(num_plots / ncols)

    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows), dpi=40)

    # Handle cases where axs might be a single Axes object if nrows=ncols=1
    if num_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten() # Flatten the 2D array of axes for easy indexing

    for i, (a, b, prop) in enumerate(zip(a_list, b_list, props)):
        ax = axs[i] # Get the current subplot axis
        y = beta.pdf(x, a, b)
        ax.plot(x, y, prop, label='a = {}, b = {}'.format(a, b))

        ax.set_xlabel('$x$')
        ax.set_ylabel('$p(x)$')
        ax.legend(loc='best', prop={'size': 8})
        ax.set_title('Beta Distribution (a={}, b={})'.format(a, b))
        sns.despine(ax=ax) # Despine for the current subplot

    # Turn off any unused subplots
    for j in range(num_plots, nrows * ncols):
        if j < len(axs):
            fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()

x = jnp.linspace(0, 1, 100)
a_list = [0.1, 0.1, 1.0, 2.0, 2.0, 5.0, 1.0, 2.0, 0.5]
b_list = [0.1, 1.0, 0.1, 0.1, 8.0, 1.0, 5.0, 2.0, 0.5]
props = ['b', 'r', 'k', 'g', 'c', 'm', 'y', 'tab:orange', 'tab:purple']
data = {'x': x, 'a_list': a_list, 'b_list': b_list, 'props':props}
graph(data)

"""Bimodal Distribution"""

def graph(ax, data, color=None, linestyle=None, label=None, xlabel=None, ylabel=None):
    line_width = 2
    x = data['x']
    weights = data['weights']
    distributions = data['distributions']
    p = sum(weights[i] * distributions[i].pdf(x) for i in range(len(distributions)) )

    ax.plot(x, p, color=color, linestyle=linestyle, linewidth=line_width, label=label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if xlabel: plt.xlabel('$x$')
    if ylabel: plt.ylabel('$p(x)$')

''' define distributions '''
data = dict()
mu = [0, 2]
sigma = [0.5, 0.5]
distributions = [norm(loc=mu[i], scale=sigma[i]) for i in range(2)]
x = jnp.linspace(-2, 2*mu[1], 600)
#p = sum(weights[i] * distributions[i].pdf(x) for i in range(2) )
weights = [0.5, 0.5]
data = {'distributions': distributions, 'weights': weights, 'x': x}

plt.figure(figsize=(10,5), dpi=60)
ax = plt.gca()
#graph(ax, data)

mu = [0]
sigma = [0.5]
distributions = [norm(loc=mu[i], scale=sigma[i]) for i in range(1)]
data1 = {'distributions': distributions, 'weights': weights, 'x': x}
graph(ax, data1, color='g', linestyle='dashdot', label='1st dist')

mu = [2]
sigma = [0.5]
distributions = [norm(loc=mu[i], scale=sigma[i]) for i in range(1)]
data2 = {'distributions': distributions, 'weights': weights, 'x': x}
graph(ax, data2, color='r', linestyle='dashdot', label='2nd dist')

graph(ax, data, color='b', linestyle='dashed', label='1st dist + 2nd dist', xlabel='$x$', ylabel='$p(x)$')

"""Central Limit Theorem"""

def plot_convolutionHist(out, N, sample_size, bins):
    counts, nbins_loc = jnp.histogram(out, bins=bins)
    counts = counts/(sample_size/bins)

    plt.figure(figsize=(10,5), dpi=60)
    plt.hist(nbins_loc[:-1], counts, width=0.02, color='black')
    plt.xticks(jnp.linspace(0, 1, 3))
    plt.yticks(jnp.linspace(0, 3, 4))
    plt.xlim(0, 1)
    plt.ylim(0, 3)
    plt.title(f'N = {N}')
    plt.xlabel('$x$')
    plt.ylabel('$freq mu$')
    sns.despine()
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp

# Modified plot_convolutionHist to accept an axes object
def plot_convolutionHist(ax, out, N, sample_size, bins):
    counts, nbins_loc = jnp.histogram(out, bins=bins)
    counts = counts / (sample_size / bins)

    # Plot directly on the provided axes
    ax.bar(nbins_loc[:-1], counts, width=0.02, align='edge', color='black')
    ax.set_xticks(jnp.linspace(0, 1, 3))
    ax.set_yticks(jnp.linspace(0, 3, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 3)
    ax.set_title(f'N = {N}')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$freq mu$')
    sns.despine(ax=ax) # Despine for the current subplot

key = jax.random.PRNGKey(1)
keys = jax.random.split(key, num=10000)
sample_size = 10000
bins = 20
N_array = [1, 5]

def mean_(keys, N):
    x = jnp.mean(jax.random.beta(keys, 1, 5, [1, N]))
    return x

# Create a single figure with subplots
fig, axs = plt.subplots(1, len(N_array), figsize=(12, 5), dpi=50) # 1 row, len(N_array) columns

for i, N in enumerate(N_array):
    means = jax.vmap(mean_, in_axes=(0, None), out_axes=0)
    out = means(keys, N)
    # Pass the specific subplot axis to the function
    plot_convolutionHist(axs[i], out, N, sample_size, bins)

plt.tight_layout() # Adjust layout to prevent overlapping titles/labels
plt.show()

"""Robust pdf"""

from scipy.stats import norm, laplace, t

outlier_pos = 0
outliers = []
samples_norm_dist = 30
samples_graph_xaxis = 500
ranges_xaxis = [-5, 10]
ranges_yaxis = [0, 0.5]
norm_dist_sample = random.normal(random.PRNGKey(42), shape=(10000,))
x_axis = jnp.linspace(-5, 5, 1000)
samples = jnp.hstack((norm_dist_sample, jnp.array(outliers)))
fig, ax = plt.subplots(figsize=(10, 5), dpi=60)

samples = norm_dist_sample
ax.hist(
    np.array(norm_dist_sample),
    bins=40,
    density=True,
    color='steelblue',
    ec='steelblue',
    weights=np.ones(len(norm_dist_sample)) / len(norm_dist_sample),
    rwidth=0.8,
    label='Normal Distribution'
)

loc, scale = norm.fit(samples)
norm_pdf = norm.pdf(x_axis, loc=loc, scale=scale)

loc, scale = laplace.fit(samples)
laplace_pdf = laplace.pdf(x_axis, loc=loc, scale=scale)

# Fix: t.fit returns (df, loc, scale), so unpack into 3 variables.
# Assuming the first variable 'df_fitted' is the degrees of freedom.
df_fitted, loc, scale = t.fit(samples, floc=0)
studentT_pdf = t.pdf(x_axis, df_fitted, loc=loc, scale=scale)

ax.plot(x_axis, norm_pdf, color='red', label='Fitted Normal Distribution')
ax.plot(x_axis, laplace_pdf, color='green', label='Fitted Laplace Distribution')
ax.plot(x_axis, studentT_pdf, color='orange', label='Fitted Student-t Distribution')

ax.set_xlabel('Value')
ax.set_ylabel('Probability Density')
ax.set_title('Distribution Comparison')
ax.legend(('gaussian', 'student-t', 'laplace', 'data'))
sns.despine()
plt.show()

"""Beta binomial model"""

import math
import scipy.stats as stats

def posterior_grid(heads, tails, grid_points=100):
    grid = np.linspace(0, 1, grid_points)
    prior = np.repeat(1/grid_points, grid_points)
    likelihood = stats.binom.pmf(heads, heads +tails, grid)
    posterior = prior * likelihood
    posterior = posterior / posterior.sum()
    return grid, posterior

data = np.repeat([0,1], [10,1])
h = data.sum()
t = len(data) - h
x = np.linspace(0, 1, 100)
xs = x
dx_exact = xs[1] - xs[0]
post_exact = stats.beta.pdf(xs, a=h+1, b=t+1)
post_exact = post_exact / np.sum(post_exact)
plt.figure(figsize=(10,5), dpi=50)
plt.plot(xs, post_exact)
plt.yticks([])
plt.title('exact posterior')
plt.show()

n = 20
grid, posterior = posterior_grid(h, t, n)
dx_grid = grid[1] - grid[0]
sf = dx_grid / dx_exact

plt.figure(figsize=(10,5), dpi=60)
plt.bar(grid, posterior, width=1/n, alpha=.2)
plt.plot(xs, post_exact * xs)
plt.xlabel('theta')
plt.yticks([])
plt.title('approximation')
plt.show()

with pm.Model() as normal_approximation:
    theta = pm.Beta('theta', 1.0, 1.0)
    y = pm.Binomial('y', n=1, p=theta, observed=data)
    mean_q = pm.find_MAP()
    std_q = ((1/pm.find_hessian(mean_q, vars=[theta]))**0.5)[0]
    mu = mean_q['theta']

plt.figure(figsize=(10,5), dpi=50)
plt.plot(xs, stats.norm.pdf(xs, mu, std_q), linestyle='--', label='Laplace')
post_exact = stats.beta.pdf(xs, a=h+1, b=t+1)
plt.plot(xs, post_exact, label='exact')
plt.xlabel('theta', fontsize=14)
plt.yticks([])
plt.title('quadratic approximation')
plt.legend()
plt.show()

with pm.Model() as hmc_model:
    theta = pm.Beta('theta', 1.0,1.0)
    y = pm.Binomial('y', n=1, p=theta, observed=data)
    trace = pm.sample(1000, random_seed=42, cores=1, chains=2)

plt.figure(figsize=(10,5), dpi=40)
# Correct way to plot posterior using arviz from InferenceData object
# 'ax' was not defined, so directly using az.plot_posterior
az.plot_posterior(trace, var_names=['theta'], hdi_prob=0.95)
plt.show()

plt.figure(figsize=(10, 6), dpi=60)
az.plot_trace(trace, var_names=['theta'])
plt.tight_layout()
plt.show()

with pm.Model() as mf_model:
    theta = pm.Beta('theta', 1.0, 1.0)
    y = pm.Binomial('y', n=1, p=theta, observed=data)
    mean_field = pm.fit(method='advi')
    trace_mf = mean_field.sample(1000)

thetas = trace_mf.posterior['theta']
axes = az.plot_posterior(thetas, hdi_prob=0.95)
plt.show()

from scipy.stats import gaussian_kde

plt.figure(figsize=(12, 6), dpi=60)

# Exact posterior
plt.plot(xs, post_exact, label='Exact Posterior', color='blue')

# Grid approximation
plt.bar(grid, posterior, width=1/n, alpha=.2, color='red', label='Grid Approximation')

# Quadratic approximation (Laplace)
plt.plot(xs, stats.norm.pdf(xs, mu, std_q), linestyle='--', color='green', label='Laplace Approximation')

# HMC sampling results (KDE of samples)
hmc_samples = trace.posterior['theta'].values.flatten()
kde = gaussian_kde(hmc_samples)
plt.plot(xs, kde(xs), linestyle=':', color='purple', label='HMC Samples (KDE)')

plt.xlabel('theta', fontsize=14)
plt.yticks([])
plt.title('Comparison of All Approximations')
plt.legend()
plt.show()

with pm.Model() as mf_model:
    theta = pm.Beta('theta', 1.0, 1.0)
    y = pm.Binomial('y', n=1, p=theta, observed=data)
    advi = pm.ADVI()
    tracker = pm.callbacks.Tracker(mean=advi.approx.mean.eval, std=advi.approx.std.eval)
    approx = advi.fit(n=1000, callbacks=[tracker])

trace_approx = approx.sample(1000)
theta = trace_approx.posterior['theta']

plt.figure(figsize=(10, 6), dpi=60)
plt.plot(tracker['mean'])
plt.plot(tracker['std'])
#plt.plot(advi.hist)
#sns.kdeplot(thetas)
plt.title('Mean - Std - negative Elbo' )

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
mu_ax = axs[0]
std_ax = axs[1]
elbo_ax = axs[2]
#kde_ax = axs[3]
mu_ax.plot(tracker['mean'])
std_ax.plot(tracker['std'])
elbo_ax.plot(advi.hist)
elbo_ax.set_title('negative elbo')
#kde_ax = sns.kdeplot(thetas)
#kde_ax.set_title('kde of posterior samples')
plt.show()

fig = plt.figure(figsize=(16, 9))
mu_ax = fig.add_subplot(2, 2, 1)
std_ax = fig.add_subplot(2, 2, 2)
hist_ax = fig.add_subplot(2, 1, 2)
mu_ax.plot(tracker['mean'])
mu_ax.set_title('Mean')
std_ax.plot(tracker['std'])
std_ax.set_title('Std')
hist_ax.plot(advi.hist)
hist_ax.set_title('negative elbo')

trace_approx = approx.sample(1000)
thetas = trace_approx.posterior['theta']
axes = az.plot_posterior(thetas, hdi_prob=0.95)
plt.show()