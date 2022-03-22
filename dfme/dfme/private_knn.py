import numpy as np
import math
import scipy
import scipy.stats

def compute_rdp_data_dependent_gnmax(logq, sigma, orders):
    """
    Computes data-dependent RDP guarantees for the GNMax mechanism.
    This is the bound D_\lambda(M(D) || M(D'))  from Theorem 6 (equation 2),
    PATE 2018 (Appendix A).

    Bounds RDP from above of GNMax given an upper bound on q.

    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax
            mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    # See Appendix A in PATE 2018.
    # rdp_order2 = sigma * math.sqrt(-logq)
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    # The GNMAx mechanism satisfies:
    # (order = \lambda, eps = \lambda / sigma^2)-RDP.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(
                1 + 1 / (rdp_order2 - 1)))):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
                log1mq - _log1mexp((logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)

        # Values of q close to 1 could result in a looser bound, so minimum
        # between the data dependent bound and the data independent bound
        # rdp_esp = orders / variance is taken.
        rdp_eps[mask] = np.minimum(rdp_eps, log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def compute_logq_gnmax(votes, sigma):
    """
    Computes an upper bound on log(Pr[outcome != argmax]) for the GNMax mechanism.

    Implementation of Proposition 7 from PATE 2018 paper.

    Args:
        votes: a 1-D numpy array of raw ensemble votes for a given query.
        sigma: std of the Gaussian noise in the GNMax mechanism.

    Returns:
        A scalar upper bound on log(Pr[outcome != argmax]) where log denotes natural logarithm.
    """
    num_classes = len(votes)
    variance = sigma ** 2
    idx_max = np.argmax(votes)
    votes_gap = votes[idx_max] - votes
    votes_gap = votes_gap[np.arange(num_classes) != idx_max]  # exclude argmax
    # Upper bound log(q) via a union bound rather than a more precise
    # calculation.
    logq = _logsumexp(
        scipy.stats.norm.logsf(votes_gap, scale=math.sqrt(2 * variance)))
    return min(logq,
               math.log(1 - (1 / num_classes)))  # another obvious upper bound


def compute_rdp_data_dependent_gnmax_no_upper_bound(logq, sigma, orders):
    """
    If the data dependent bound applies, then use it even though its higher than
    the data independent bound. In this case, we are interested in estimating
    the privacy budget solely on the data and are not optimizing its value to be
    as small as possible.

    Computes data-dependent RDP guarantees for the GNMax mechanism.
    This is the bound D_\lambda(M(D) || M(D'))  from Theorem 6 (equation 2),
    PATE 2018 (Appendix A).

    Bounds RDP from above of GNMax given an upper bound on q.

    Args:
        logq: a union bound on log(Pr[outcome != argmax]) for the GNMax
            mechanism.
        sigma: std of the Gaussian noise in the GNMax mechanism.
        orders: an array-like list of RDP orders.

    Returns:
        A numpy array of upper bounds on RDP for all orders.

    Raises:
        ValueError: if the inputs are invalid.
    """
    if logq > 0 or sigma < 0 or np.isscalar(orders) or np.any(orders <= 1):
        raise ValueError(
            "'logq' must be non-positive, 'sigma' must be non-negative, "
            "'orders' must be array-like, and all elements in 'orders' must be "
            "greater than 1!")

    if np.isneginf(logq):  # deterministic mechanism with sigma == 0
        return np.full_like(orders, 0., dtype=np.float)

    variance = sigma ** 2
    orders = np.array(orders)
    rdp_eps = orders / variance  # data-independent bound as baseline

    # Two different higher orders computed according to Proposition 10.
    # See Appendix A in PATE 2018.
    # rdp_order2 = sigma * math.sqrt(-logq)
    rdp_order2 = math.sqrt(variance * -logq)
    rdp_order1 = rdp_order2 + 1

    # Filter out entries to which data-dependent bound does not apply.
    mask = np.logical_and(rdp_order1 > orders, rdp_order2 > 1)

    # Corresponding RDP guarantees for the two higher orders.
    # The GNMAx mechanism satisfies:
    # (order = \lambda, eps = \lambda / sigma^2)-RDP.
    rdp_eps1 = rdp_order1 / variance
    rdp_eps2 = rdp_order2 / variance

    log_a2 = (rdp_order2 - 1) * rdp_eps2

    # Make sure that logq lies in the increasing range and that A is positive.
    if (np.any(mask) and -logq > rdp_eps2 and logq <= log_a2 - rdp_order2 *
            (math.log(1 + 1 / (rdp_order1 - 1)) + math.log(
                1 + 1 / (rdp_order2 - 1)))):
        # Use log1p(x) = log(1 + x) to avoid catastrophic cancellations when x ~ 0.
        log1mq = _log1mexp(logq)  # log1mq = log(1-q)
        log_a = (orders - 1) * (
                log1mq - _log1mexp(
            (logq + rdp_eps2) * (1 - 1 / rdp_order2)))
        log_b = (orders - 1) * (rdp_eps1 - logq / (rdp_order1 - 1))

        # Use logaddexp(x, y) = log(e^x + e^y) to avoid overflow for large x, y.
        log_s = np.logaddexp(log1mq + log_a, logq + log_b)

        # Do not apply the minimum between the data independent and data
        # dependent bound - but limit the computation to data dependent bound
        # only!
        rdp_eps[mask] = (log_s / (orders - 1))[mask]

    assert np.all(rdp_eps >= 0)
    return rdp_eps


def rdp_to_dp(orders, rdp_eps, delta):
    """
    Conversion from (lambda, eps)-RDP to conventional (eps, delta)-DP.
    Papernot 2018, Theorem 5. (From RDP to DP)

    Args:
        orders: an array-like list of RDP orders.
        rdp_eps: an array-like list of RDP guarantees (of the same length as
        orders).
        delta: target delta (a scalar).

    Returns:
        A pair of (dp_eps, optimal_order).
    """
    assert not np.isscalar(orders) and not np.isscalar(rdp_eps) and len(
        orders) == len(
        rdp_eps), "'orders' and 'rdp_eps' must be array-like and of the same length!"

    dp_eps = np.array(rdp_eps) - math.log(delta) / (np.array(orders) - 1)
    idx_opt = np.argmin(dp_eps)
    return dp_eps[idx_opt], orders[idx_opt]

def _logsumexp(x):
    """
    Sum in the log space.

    An addition operation in the standard linear-scale becomes the
    LSE (log-sum-exp) in log-scale.

    Args:
        x: array-like.

    Returns:
        A scalar.
    """
    x = np.array(x)
    m = max(x)  # for numerical stability
    return m + math.log(sum(np.exp(x - m)))

def _log1mexp(x):
    """
    Numerically stable computation of log(1-exp(x)).

    Args:
        x: a scalar.

    Returns:
        A scalar.
    """
    assert x <= 0, "Argument must be positive!"
    # assert x < 0, "Argument must be non-negative!"
    if x < -1:
        return math.log1p(-math.exp(x))
    elif x < 0:
        return math.log(-math.expm1(x))
    else:
        return -np.inf

class PrivateKnn:
    """
    Compute the privacy budget based on Private kNN version of PATE.
    Find the neighbors of a new data point among the training points in the
    representation from the last layer. The neighbors are teachers who vote
    with their ground-truth labels to create the histogram of votes for PATE.
    """

    def __init__(self, delta, sigma_gnmax, apply_data_independent_bound=False):
        """
        Initialize the stateful private knn to keep track of the privacy cost.
        Args:
            delta: pre-defined delta value for (eps, delta)-DP. A commonly used
                value is the inverse of number of the training points.
            sigma_gnmax: std of the Gaussian noise for the DP mechanism.
        """
        self.delta = delta
        self.sigma_gnmax = sigma_gnmax
        self.apply_data_independent_bound = apply_data_independent_bound

        # RDP orders.
        self.orders = np.concatenate(
            (
                np.arange(2, 100, .5),
                np.logspace(np.log10(100), np.log10(1000), num=200),
            )
        )

        # Current cumulative results
        self.rdp_eps_total_curr = np.zeros(len(self.orders))

    def _compute_partition(self, order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(self.orders, order_opt)
        rdp_eps_gnmax = self.rdp_eps_total_curr[idx]
        p = np.array([rdp_eps_gnmax, -math.log(self.delta) / (order_opt - 1)])
        assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    def add_privacy_cost(self, votes):
        """
        Analyze and compute the additional privacy cost incurred when
        answering these additional queries using the gaussian noisy max
        algorithm but without the thresholding mechanism.
        Args:
            votes: a 2-D numpy array of raw ensemble votes, with each row
            corresponding to a query.
        Returns:
            dp_eps: a numpy array of length L = num-queries, with each entry
                corresponding to the privacy cost at a specific moment.
        """
        # Number of new queries.
        n = votes.shape[0]

        # Iterating over all queries
        for i in range(n):
            v = votes[i]
            logq = compute_logq_gnmax(v, self.sigma_gnmax)
            if self.apply_data_independent_bound:
                rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
                    logq, self.sigma_gnmax, self.orders)
            else:
                rdp_eps_gnmax = compute_rdp_data_dependent_gnmax_no_upper_bound(
                    logq, self.sigma_gnmax, self.orders)

            # Update current cumulative results.
            self.rdp_eps_total_curr += rdp_eps_gnmax

        return self.get_current_dp_eps()

    def get_current_dp_eps(self):
        """
        Returns: current cumulative epsilon for DP(epsilon,delta) computed on
        all the queries seen so far.
        """
        dp_eps, _ = rdp_to_dp(
            self.orders, self.rdp_eps_total_curr, self.delta)
        return dp_eps