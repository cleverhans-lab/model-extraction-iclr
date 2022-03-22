from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import sys

from analysis.pate import compute_logpr_answered
from analysis.pate import compute_logq_gnmax
from analysis.pate import compute_logq_multilabel_pate
from analysis.pate import compute_rdp_data_dependent_gnmax
from analysis.pate import compute_rdp_data_dependent_gnmax_no_upper_bound
from analysis.pate import compute_rdp_data_dependent_threshold
from analysis.pate import compute_rdp_data_independent_multilabel
from analysis.pate import rdp_to_dp
from utils import augmented_print


def analyze_results(votes, max_num_query, dp_eps):
    print('max_num_query;', max_num_query)
    dp_eps_items = []
    # eps were added to the sum of previous epsilons - subtract the value
    # to get single epsilons.
    dp_eps_items.append(dp_eps[0])
    for i in range(1, len(dp_eps)):
        dp_eps_items.append(dp_eps[i] - dp_eps[i - 1])
    dp_eps_items = np.array(dp_eps_items)
    avg_dp_eps = np.mean(dp_eps_items)
    print('avg_dp_eps;', avg_dp_eps)
    print('min_dp_eps;', np.min(dp_eps_items))
    print('median_dp_eps;', np.median(dp_eps_items))
    print('mean_dp_eps;', np.mean(dp_eps_items))
    print('max_dp_eps;', np.max(dp_eps_items))
    print('sum_dp_eps;', np.sum(dp_eps_items))
    print('std_dp_eps;', np.std(dp_eps_items))

    # Sort votes in ascending orders.
    sorted_votes = np.sort(votes, axis=-1)
    # Subtract runner-up votes from the max number of votes.
    gaps = sorted_votes[:, -1] - sorted_votes[:, -2]

    assert np.all(gaps > 0)
    print('min gaps;', np.min(gaps))
    print('avg gaps;', np.mean(gaps))
    print('median gaps;', np.median(gaps))
    print('max gaps;', np.max(gaps))
    print('sum gaps;', np.sum(dp_eps_items))
    print('std gaps;', np.std(dp_eps_items))

    # aggregate
    unique_gaps = np.unique(np.sort(gaps))
    gap_eps = {}
    print('gap;mean_eps')
    for gap in unique_gaps:
        mean_eps = dp_eps_items[gaps == gap].mean()
        gap_eps[gap] = mean_eps
        print(f'{gap};{mean_eps}')

    return gap_eps, gaps


def analyze_multiclass_confident_gnmax(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta, file,
        show_dp_budget='disable', args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the Confident GNMax mechanism.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry
            corresponding to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the expected number of answered queries at a
            specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry
            corresponding to the order minimizing the privacy cost at a
            specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_threshold = rdp_eps_threshold_curr[idx]
        rdp_eps_gnmax = rdp_eps_total_curr[idx] - rdp_eps_threshold
        p = np.array([rdp_eps_threshold, rdp_eps_gnmax,
                      -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = votes.shape[0]
    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)
    answered = np.zeros(n, dtype=float)
    # Current cumulative results
    rdp_eps_threshold_curr = np.zeros(len(orders))
    rdp_eps_total_curr = np.zeros(len(orders))
    rdp_eps_total_sqrd_curr = np.zeros(len(orders))
    answered_curr = 0
    # Iterating over all queries
    for i in range(n):
        v = votes[i]
        if sigma_threshold > 0:
            # logpr - probability that the label is answered.
            logpr = compute_logpr_answered(threshold, sigma_threshold, v)
            rdp_eps_threshold = compute_rdp_data_dependent_threshold(
                logpr, sigma_threshold, orders)
        else:
            # Do not use the Confident part of the GNMax.
            assert threshold == 0
            logpr = 0
            rdp_eps_threshold = 0

        logq = compute_logq_gnmax(v, sigma_gnmax)
        rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
            logq, sigma_gnmax, orders)
        rdp_eps_total = rdp_eps_threshold + np.exp(logpr) * rdp_eps_gnmax
        # Evaluate E[(rdp_eps_threshold + Bernoulli(pr) * rdp_eps_gnmax)^2]
        rdp_eps_total_sqrd = (
                rdp_eps_threshold ** 2 + 2 * rdp_eps_threshold * np.exp(
            logpr) * rdp_eps_gnmax + np.exp(logpr) * rdp_eps_gnmax ** 2)
        # Update current cumulative results.
        rdp_eps_threshold_curr += rdp_eps_threshold
        rdp_eps_total_curr += rdp_eps_total
        rdp_eps_total_sqrd_curr += rdp_eps_total_sqrd
        pr_answered = np.exp(logpr)
        answered_curr += pr_answered
        # Update all cumulative results.
        answered[i] = answered_curr
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
        # Logs
        # if i % 100000 == 0 and i > 0:
        if show_dp_budget == 'apply':
            file = f'queries_answered_privacy_budget.txt'
            with open(file, 'a') as writer:
                if i == 0:
                    header = "queries answered,privacy budget"
                    print(header)
                    writer.write(f"{header}\n")
                info = f"{answered_curr},{dp_eps[i]}"
                print(info)
                writer.write(f"{info}\n")
                print(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                writer.write(
                    'Number of queries: {} | E[answered]: {:.3f} | E[eps] at order {:.3f}: {:.4f} (contribution from delta: {:.4f})\n'.format(
                        i + 1, answered_curr, order_opt[i], dp_eps[i],
                        -math.log(delta) / (order_opt[i] - 1)))
                sys.stdout.flush()
                writer.flush()

    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_multilabel_tau_data_independent(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta, file,
        show_dp_budget='disable', args=None):
    """
     Analyze how the pre-defined privacy budget will be exhausted when answering
     queries using the tau-approximation (clipping mechanism) for the multilabel
     classification.

     Args:
         votes: a 2-D numpy array of raw ensemble votes, with each row
         corresponding to a query.
         threshold: not used but for compatibility with confident gnmax it
             is here
         sigma_threshold: not used but for compatibility is here
         sigma_gnmax: std of the Gaussian noise for the DP mechanism.
         budget: pre-defined epsilon value for (eps, delta)-DP.
         delta: pre-defined delta value for (eps, delta)-DP.
         file: for logs.
         show_dp_budget: show the current cumulative dp budget.
         args: all args of the program

     Returns:
         max_num_query: when the pre-defined privacy budget is exhausted.
         dp_eps: a numpy array of length L = num-queries, with each entry corresponding
             to the privacy cost at a specific moment.
         partition: a numpy array of length L = num-queries, with each entry corresponding
             to the partition of privacy cost at a specific moment.
         answered: a numpy array of length L = num-queries, with each entry corresponding
             to the expected number of answered queries at a specific moment.
         order_opt: a numpy array of length L = num-queries, with each entry corresponding
             to the order minimizing the privacy cost at a specific moment.
     """
    assert args is not None
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_gnmax = rdp_eps_total_curr[idx]
        p = np.array([rdp_eps_gnmax, -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = votes.shape[0]

    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)

    # Current cumulative results
    rdp_eps_total_curr = np.zeros(len(orders))
    # Iterating over all queries
    for i in range(n):
        rdp_eps = compute_rdp_data_independent_multilabel(
            sigma=sigma_gnmax, orders=orders, tau=args.private_tau,
            norm=args.private_tau_norm)
        # Update current cumulative results.
        rdp_eps_total_curr += rdp_eps
        # Update all cumulative results.
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
        # Logs
        # if i % 100000 == 0 and i > 0:
        if show_dp_budget == 'apply':
            raw_file = f'queries_answered_privacy_budget_multilabel_tau_pate.txt'
            with open(raw_file, 'a+') as writer:
                if i == 0:
                    header = "queries answered,privacy budget"
                    writer.write(f"{header}\n")
                    writer.write("0,0\n")
                info = f"{i + 1},{dp_eps[i]}"
                writer.write(f"{info}\n")

    if file and args.debug is True:
        with open('privacy_budget_analysis_multilabel_tau_pate.csv',
                  'a+') as writer:
            info = f"{n},{dp_eps[n - 1]}"
            writer.write(f"{info}\n")

    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    # answered = [x for x in range(1, max_num_query + 1)]
    # answered is the probability of a given label being answered. For the GNMax
    # without the confidence (no thresholding mechanism) each
    # label < max_num_query is answered.
    answered = np.zeros(n, dtype=float)
    answered[0:max_num_query] = 1
    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_multiclass_gnmax(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta,
        file=None, show_dp_budget='disable', args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the gaussian noisy max algorithm but without the
    thresholding mechanism.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: not used but for compatibility with confident gnmax it
            is here
        sigma_threshold: not used but for compatibility is here
        sigma_gnmax: std of the Gaussian noise for the DP mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry corresponding
            to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry corresponding
            to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry corresponding
            to the expected number of answered queries at a specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry corresponding
            to the order minimizing the privacy cost at a specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_gnmax = rdp_eps_total_curr[idx]
        p = np.array([rdp_eps_gnmax, -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = votes.shape[0]

    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)

    # Current cumulative results
    rdp_eps_total_curr = np.zeros(len(orders))
    # Iterating over all queries
    for i in range(n):
        v = votes[i]
        logq = compute_logq_gnmax(v, sigma_gnmax)
        if args.apply_data_independent_bound:
            rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
                logq, sigma_gnmax, orders)
        else:
            rdp_eps_gnmax = compute_rdp_data_dependent_gnmax_no_upper_bound(
                logq, sigma_gnmax, orders)

        # Update current cumulative results.
        rdp_eps_total_curr += rdp_eps_gnmax
        # Update all cumulative results.
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
        # Logs
        # if i % 100000 == 0 and i > 0:
        if show_dp_budget == 'apply':
            raw_file = f'queries_answered_privacy_budget.txt'
            with open(raw_file, 'a+') as writer:
                if i == 0:
                    header = "queries answered,privacy budget"
                    writer.write(f"{header}\n")
                    writer.write("0,0\n")
                info = f"{i + 1},{dp_eps[i]}"
                writer.write(f"{info}\n")

    if file is not None:
        with open('privacy_budget_analysis.csv', 'a+') as writer:
            info = f"{n},{dp_eps[n - 1]}"
            writer.write(f"{info}\n")

    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    # answered = [x for x in range(1, max_num_query + 1)]
    # answered is the probability of a given label being answered. For the GNMax
    # without the confidence (no thresholding mechanism) each
    # label < max_num_query is answered.
    answered = np.zeros(n, dtype=float)
    answered[0:max_num_query] = 1
    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_multilabel_pate(
        votes, threshold, sigma_threshold, sigma_gnmax, budget, delta, file,
        show_dp_budget='disable', args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the gaussian noisy max algorithm but without the
    thresholding mechanism and for the multilabel task.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
        corresponding to a query.
        threshold: not used but for compatibility with confident gnmax it
            is here
        sigma_threshold: not used but for compatibility is here
        sigma_gnmax: std of the Gaussian noise for the DP mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry corresponding
            to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry corresponding
            to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry corresponding
            to the expected number of answered queries at a specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry corresponding
            to the order minimizing the privacy cost at a specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_gnmax = rdp_eps_total_curr[idx]
        p = np.array([rdp_eps_gnmax, -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    n = votes.shape[0]

    # All cumulative results
    dp_eps = np.zeros(n)
    partition = [None] * n
    order_opt = np.full(n, np.nan, dtype=float)

    # Current cumulative results
    rdp_eps_total_curr = np.zeros(len(orders))
    # Iterating over all queries
    for i in range(n):
        v = votes[i]
        logq = compute_logq_multilabel_pate(v, sigma_gnmax)
        rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
            logq, sigma_gnmax, orders)
        # Update current cumulative results.
        rdp_eps_total_curr += rdp_eps_gnmax
        # Update all cumulative results.
        dp_eps[i], order_opt[i] = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break
    # if file:
    #     with open(file, 'a+') as writer:
    #         info = f"{n},{dp_eps[n - 1]}"
    #         writer.write(f"{info}\n")
    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    answered = [x for x in range(1, max_num_query + 1)]
    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_tau_pate(votes, threshold, sigma_threshold, sigma_gnmax, budget,
                     delta, file, args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    multilabel queries using the Confident GNMax mechanism with the per label
    and per query bounds (the new addition is the per query bound).

    Args:
        votes: a 3-D numpy array of raw ensemble votes, with each entry in 2nd
            dimension corresponding to a query, and the last dimension are the
            votes for the binary classification.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        show_dp_budget: show the current cumulative dp budget.
        args: all args of the program

    Returns:
        max_num_query: max number of query answered when the pre-defined
            privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry
            corresponding to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry
            corresponding to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry
            corresponding to the expected number of answered queries at a
            specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry
            corresponding to the order minimizing the privacy cost at a
            specific moment.
    """
    max_num_query = 0

    def compute_partition(order_opt, eps):
        """Analyze how the current privacy cost is divided."""
        idx = np.searchsorted(orders, order_opt)
        rdp_eps_threshold = rdp_eps_threshold_curr[idx]
        rdp_eps_gnmax = rdp_eps_total_curr[idx] - rdp_eps_threshold
        p = np.array([rdp_eps_threshold, rdp_eps_gnmax,
                      -math.log(delta) / (order_opt - 1)])
        # assert sum(p) == eps
        # Normalize p so that sum(p) = 1
        return p / eps

    # RDP orders.
    orders = np.concatenate((np.arange(2, 100, .5),
                             np.logspace(np.log10(100), np.log10(1000),
                                         num=200)))
    # Number of queries
    num_queries = votes.shape[0]
    num_labels = votes.shape[1]

    # All cumulative results
    dp_eps = np.zeros(num_queries)
    partition = [None] * num_queries
    order_opt = np.full(num_queries, np.nan, dtype=float)
    answered = np.zeros(num_queries, dtype=float)

    # Current cumulative results
    rdp_eps_threshold_curr = np.zeros(len(orders))
    rdp_eps_total_curr = np.zeros(len(orders))
    rdp_eps_total_sqrd_curr = np.zeros(len(orders))
    answered_queries = 0

    variance = sigma_gnmax ** 2
    tau = args.private_tau
    if tau is not None and args.private_tau_norm == '2':
        # data-independent bound per query as a baseline
        rdp_eps_bound_query = tau ** 2 * orders / variance
    else:
        rdp_eps_bound_query = num_labels * orders / variance

    # Iterating over all queries.
    for i in range(num_queries):
        # Query cumulative results.
        rdp_eps_threshold_query = np.zeros(len(orders))
        rdp_eps_total_query = np.zeros(len(orders))
        rdp_eps_total_sqrd_query = np.zeros(len(orders))
        answered_labels = 0

        for j in range(num_labels):
            v = votes[i][j]
            if sigma_threshold > 0:
                # logpr - probability that the label is answered.
                logpr = compute_logpr_answered(threshold, sigma_threshold, v)
                rdp_eps_threshold = compute_rdp_data_dependent_threshold(
                    logpr, sigma_threshold, orders)
            else:
                # Do not use the Confident part of the GNMax.
                assert threshold == 0
                logpr = 0
                rdp_eps_threshold = 0
            logq = compute_logq_gnmax(v, sigma_gnmax)
            rdp_eps_gnmax = compute_rdp_data_dependent_gnmax(
                logq, sigma_gnmax, orders)
            rdp_eps_total = rdp_eps_threshold + np.exp(logpr) * rdp_eps_gnmax
            # Evaluate E[(rdp_eps_threshold + Bernoulli(pr) * rdp_eps_gnmax)^2]
            rdp_eps_total_sqrd = (
                    rdp_eps_threshold ** 2 + 2 * rdp_eps_threshold * np.exp(
                logpr) * rdp_eps_gnmax + np.exp(logpr) * rdp_eps_gnmax ** 2)

            # Update query cumulative results.
            rdp_eps_threshold_query += rdp_eps_threshold
            rdp_eps_total_query += rdp_eps_total
            rdp_eps_total_sqrd_query += rdp_eps_total_sqrd
            pr_answered_label = np.exp(logpr)
            answered_labels += pr_answered_label

        # Apply the upper data-independent bound per query.
        mask = (rdp_eps_bound_query < rdp_eps_total_query)
        rdp_eps_total_query[mask] = rdp_eps_bound_query[mask]
        rdp_eps_threshold_query[mask] = 0
        rdp_eps_total_sqrd_query[mask] = rdp_eps_bound_query[mask] ** 2

        # Update current cumulative results.
        rdp_eps_threshold_curr += rdp_eps_threshold_query
        rdp_eps_total_curr += rdp_eps_total_query
        rdp_eps_total_sqrd_curr += rdp_eps_total_sqrd_query
        answered_queries += (answered_labels / float(num_labels))

        # Update all cumulative results.
        answered[i] = answered_queries
        eps, opt = rdp_to_dp(orders, rdp_eps_total_curr, delta)
        dp_eps[i] = eps
        order_opt[i] = opt
        partition[i] = compute_partition(order_opt[i], dp_eps[i])
        # Verify if the pre-defined privacy budget is exhausted.
        if dp_eps[i] <= budget:
            max_num_query = i + 1
        else:
            break

    # print(f"{threshold},{sigma_threshold},{sigma_gnmax}")
    # analyze_results(votes=votes, max_num_query=max_num_query, dp_eps=dp_eps)
    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_multilabel(votes, threshold, sigma_threshold, sigma_gnmax, budget,
                       delta, file, args=None):
    """
    Analyze how the pre-defined privacy budget will be exhausted when answering
    queries using the (Confident) GNMax mechanism.

    Args:
        votes: a 2-D numpy array of raw ensemble votes, with each row
            corresponding to a query.
        threshold: threshold value (a scalar) in the threshold mechanism.
        sigma_threshold: std of the Gaussian noise in the threshold mechanism.
        sigma_gnmax: std of the Gaussian noise in the GNMax mechanism.
        budget: pre-defined epsilon value for (eps, delta)-DP.
        delta: pre-defined delta value for (eps, delta)-DP.
        file: for logs.
        args: all args of the program

    Returns:
        max_num_query: when the pre-defined privacy budget is exhausted.
        dp_eps: a numpy array of length L = num-queries, with each entry corresponding
            to the privacy cost at a specific moment.
        partition: a numpy array of length L = num-queries, with each entry corresponding
            to the partition of privacy cost at a specific moment.
        answered: a numpy array of length L = num-queries, with each entry corresponding
            to the expected number of answered queries at a specific moment.
        order_opt: a numpy array of length L = num-queries, with each entry corresponding
            to the order minimizing the privacy cost at a specific moment.
    """
    # augmented_print(
    #     text="Make the all the votes expressed as labels into a single "
    #          "dimensional array.",
    #     file=file)
    num_queries = votes.shape[0]
    num_labels = votes.shape[1]
    assert votes.shape[2] == 2

    # augmented_print(text=f"number of queries: {num_queries}", file=file)
    # augmented_print(text=f"number of labels: {num_labels}", file=file)

    votes = votes.reshape((-1, 2))

    # augmented_print(text=f"number of transformed votes: {votes.shape[0]}",
    #                 file=file)

    if threshold == 0:
        assert sigma_threshold == 0
        analyze_fun = analyze_multiclass_gnmax
    else:
        analyze_fun = analyze_multiclass_confident_gnmax

    max_num_query, dp_eps, partition, answered, order_opt = analyze_fun(
        votes=votes, threshold=threshold, sigma_threshold=sigma_threshold,
        sigma_gnmax=sigma_gnmax, budget=budget, delta=delta, file=file)

    # augmented_print("Maximum number of queries: {}".format(max_num_query), file)
    # augmented_print("Privacy guarantee achieved: ({:.4f})".format(
    #     dp_eps[max_num_query - 1]), file)
    # augmented_print("Expected number of queries answered: {:.3f}".format(
    #     answered[max_num_query - 1]), file)
    # augmented_print("Partition of privacy cost: {}".format(
    #     np.array2string(partition[max_num_query - 1], precision=3,
    #                     separator=', ')), file)
    # print('Label answered, Privacy cost, Cumulative privacy cost')
    # for i in range(max_num_query):
    #     if i == 0:
    #         cost = dp_eps[0]
    #     else:
    #         cost = dp_eps[i] - dp_eps[i - 1]
    #     print(f'{i},{cost},{dp_eps[i]}')

    max_num_query //= num_labels
    dp_eps = np.array(dp_eps).reshape((num_queries, num_labels))[:, -1]
    answered = np.array(answered).reshape((num_queries, num_labels))[:, -1]
    answered //= num_labels

    return max_num_query, dp_eps, partition, answered, order_opt


def analyze_multilabel_counting(votes, threshold, sigma_threshold, sigma_gnmax,
                                budget, delta, file, args=None):
    augmented_print(
        text="Make the all the votes expressed as labels into a single dimensional array.",
        file=file)
    num_queries = votes.shape[0]
    num_labels = votes.shape[1]
    augmented_print(text=f"number of queries: {num_queries}", file=file)
    augmented_print(text=f"number of labels: {num_labels}", file=file)

    max_num_query = num_queries
    dp_eps = np.arange(0, budget, budget / num_queries)
    partition = np.repeat(budget / num_queries, num_queries)
    answered = np.arange(0, num_queries, 1)
    order_opt = None

    return max_num_query, dp_eps, partition, answered, order_opt
