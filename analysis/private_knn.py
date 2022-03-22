import numpy as np
import math

from analysis.rdp_cumulative import compute_rdp_data_dependent_gnmax
from analysis.rdp_cumulative import compute_logq_gnmax
from analysis.rdp_cumulative import \
    compute_rdp_data_dependent_gnmax_no_upper_bound
from analysis.rdp_cumulative import rdp_to_dp


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