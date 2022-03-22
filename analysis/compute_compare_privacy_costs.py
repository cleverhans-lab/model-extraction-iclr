import math
import scipy
import scipy.stats
import numpy as np
from scipy import stats
import seaborn as sns
from analysis import pate
from collections import defaultdict
from matplotlib import pyplot as plt
import pandas as pd


def _log1mexp(x):
  """Numerically stable computation of log(1-exp(x))."""
  if x < -1:
    return math.log1p(-math.exp(x))
  elif x < 0:
    return math.log(-math.expm1(x))
  elif x == 0:
    return -np.inf
  else:
    raise ValueError("Argument must be non-positive.")

def _logaddexp(x):
  """Addition in the log space. Analogue of numpy.logaddexp for a list."""
  m = max(x)
  return m + math.log(sum(np.exp(x - m)))


def rdp_gaussian(logq, sigma, orders):
  """Bounds RDP from above of GNMax given an upper bound on q (Theorem 6).
  Args:
    logq: Natural logarithm of the probability of a non-argmax outcome.
    sigma: Standard deviation of Gaussian noise.
    orders: An array_like list of Renyi orders.
  Returns:
    Upper bound on RPD for all orders. A scalar if orders is a scalar.
  Raises:
    ValueError: If the input is malformed.
  """
  if logq > 0 or sigma < 0 or np.any(orders <= 1):  # not defined for alpha=1
    raise ValueError("Inputs are malformed.")

  if np.isneginf(logq):  # If the mechanism's output is fixed, it has 0-DP.
    if np.isscalar(orders):
      return 0.
    else:
      return np.full_like(orders, 0., dtype=np.float)

  variance = sigma ** 2

  # Use two different higher orders: mu_hi1 and mu_hi2 computed according to
  # Proposition 10.
  mu_hi2 = math.sqrt(variance * -logq)
  mu_hi1 = mu_hi2 + 1

  orders_vec = np.atleast_1d(orders)

  ret = orders_vec / variance  # baseline: data-independent bound

  # Filter out entries where data-dependent bound does not apply.
  mask = np.logical_and(mu_hi1 > orders_vec, mu_hi2 > 1)

  rdp_hi1 = mu_hi1 / variance
  rdp_hi2 = mu_hi2 / variance

  log_a2 = (mu_hi2 - 1) * rdp_hi2

  # Make sure q is in the increasing wrt q range and A is positive.
  if (np.any(mask) and logq <= log_a2 - mu_hi2 *
      (math.log(1 + 1 / (mu_hi1 - 1)) + math.log(1 + 1 / (mu_hi2 - 1))) and
      -logq > rdp_hi2):
    # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
    log1q = _log1mexp(logq)  # log1q = log(1-q)
    log_a = (orders - 1) * (
        log1q - _log1mexp((logq + rdp_hi2) * (1 - 1 / mu_hi2)))
    log_b = (orders - 1) * (rdp_hi1 - logq / (mu_hi1 - 1))

    # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
    log_s = np.logaddexp(log1q + log_a, logq + log_b)
    ret[mask] = np.minimum(ret, log_s / (orders - 1))[mask]

  assert np.all(ret >= 0)

  if np.isscalar(orders):
    return np.asscalar(ret)
  else:
    return ret


def get_logq(counts, sigma):
  n_parties = 50
  variance = sigma ** 2
  counts_normalized = np.abs(2 * counts - n_parties)
  logq = _logaddexp(scipy.stats.norm.logsf(counts_normalized, scale=np.sqrt(variance)))
  return logq


# def get_logq(counts, sigma):
#   n_parties = 50
#   variance = sigma ** 2
#   counts_normalized = np.abs(2 * counts - n_parties)
#   pos = counts_normalized[counts > 25]
#   neg = counts_normalized[counts < 25]
#
#   # print(np.exp())
#   print(scipy.stats.norm.logcdf(counts_normalized))
#   # return np.log(1)
#   # logq = _logaddexp(scipy.stats.norm.logsf(counts_normalized, scale=np.sqrt(variance)))
#   # return logq


def compute_logq_gnmax_counting(votes, sigma, thresholds):
  """
  Computes an upper bound on log(Pr[outcome != S]) for the GNMax mechanism.

  Implementation of Proposition 1.3 / 1.4 from CaPC paper.

  Args:
      votes: a 1-D numpy array of raw ensemble votes for a given query.
      sigma: std of the Gaussian noise in the GNMax mechanism.

  Returns:
      A scalar upper bound on log(Pr[outcome != S]) where log denotes natural logarithm.
  """
  num_classes = len(votes)
  variance = sigma ** 2
  vector = votes >= thresholds
  gap = votes - thresholds

  ### Flip if not in the most likely outcome S
  gap[vector == 0] = -1 * gap[vector == 0]

  logq = pate._logsumexp(
    scipy.stats.norm.logsf(gap, scale=math.sqrt(variance)))
  return logq


# for sigma in range(30, 35):
orders = np.array([2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15.,])
counts = np.array([45, 40, 10])
num_classes = len(counts)
df = defaultdict(lambda: [])

def append_df(df, **kwargs):
  for key, val in kwargs.items():
    df[key].append(val)

for sigma in range(7, 10):
  logq = get_logq(counts, sigma)
  print(logq)
  bound_loqg = math.log(1 - (1 / 2 ** (num_classes)))
  budgets = rdp_gaussian(logq, sigma, orders)
  print(f"new analysis| highest sigma: {sigma}, logq: {logq}, highest usage: {np.max(budgets)} for order: {np.argmax(budgets) + 2}")
  append_df(df, sigma=sigma, max_budget=np.max(budgets), logq=logq, type='new', queries_to_two=2./np.max(budgets))
  # logq = compute_logq_gnmax_counting(counts, sigma, [25 for _ in range(len(counts))])
  # budgets = rdp_gaussian(logq, sigma, orders)
  # print(f"old analysis_test| highest sigma: {sigma}, logq: {logq}, highest usage: {np.max(budgets)} for order: {np.argmax(budgets) + 2}")
  # budgets = rdp_gaussian(logq, sigma, orders)
  # print(f"obvious bound analysis_test| highest sigma: {sigma}, logq: {bound_loqg}, highest usage: {np.max(budgets)} for order: {np.argmax(budgets) + 2}")
  logq = compute_logq_gnmax_counting(counts, sigma, [25 for _ in range(len(counts))])
  budgets = rdp_gaussian(logq, sigma, orders)
  print(f"full analysis| highest sigma: {sigma}, logq: {logq}, highest usage: {np.max(budgets)} for order: {np.argmax(budgets) + 2}")
  append_df(df, sigma=sigma, max_budget=np.max(budgets), logq=logq, type='old', queries_to_two=2./np.max(budgets))

df = pd.DataFrame(df)

plt.figure()
sns.lineplot(x='sigma', y='max_budget', style='type', hue='type', data=df)
plt.figure()
sns.lineplot(x='sigma', y='logq', style='type', hue='type', data=df)
plt.figure()
sns.lineplot(x='sigma', y='queries_to_two', style='type', hue='type', data=df)
plt.show()