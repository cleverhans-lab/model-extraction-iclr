import numpy as np
from numpy import genfromtxt
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator


def to_csv(array: np.ndarray):
    return ",".join([str(x) for x in array])


class PoW:
    """
    PoW - Proof-of-Work.

    Translate the current privacy cost incurred by a user to an additional
    timing per query due to the proof of work.

    """

    def __init__(self, dataset='cifar10', batch_size=64, min_timing=1e-1,
                 use_exp=True, type=None):
        """
        Initialization.

        :param dataset: the name of the dataset.
        :param batch_size: the number of samples per query.
        :param min_timing: (in sec) - the minimum additional time from PoW.
        :param use_exp: use exponential growth from privacy cost to PoW timing

        PoW timings per query were computed experimentally using the hashcash
        cost function. Each i-th entry corresponds to the average timing per
        query that is an average additional time spent on solving a challenge
        with i leading (bit) zeros. The challenge is sent by the server to the
        querying user.

        Ultimately, we want to find a function that maps from a user's cost of
        queries (e.g., the incurred privacy cost) to the difficulty of the PoW
        puzzle.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.min_timing = min_timing
        self.use_exp = use_exp
        self.type = type # to run pknn

        """
        Map from the 'time required to solve the puzzle' (key) to the number 
        of leading zeros in the puzzle (value).
        """
        self.pow_timing_leading_zero_bits = {
            0.0: 0,
            0.000018: 1,
            0.000046: 2,
            0.00005: 3,
            0.000056: 4,
            0.000181: 5,
            0.000315: 6,
            0.000325: 7,
            0.000704: 8,
            0.001079: 9,
            0.003480: 10,
            0.007472: 11,
            0.008756: 12,
            0.018955: 13,
            0.049600: 14,
            0.065657: 15,
            0.239012: 16,
            0.489723: 17,
            0.549378: 18,
            1.474332: 19,
            2.927822: 20,
            4.340743: 21,
            8.519815: 22,
            15.106678: 23,
            23.196483: 24,
            151.251428: 25,
            245.364584: 26,
            565.612546: 27,
            666.044603: 28,
            1069.022753: 29,
            1533.945173: 30,
            2832.509785: 31,
            2948.467908: 32,
        }

        """
        Arrange only the estimated timing per query with PoW as a sorted array.
        """
        self.all_pow_timings_only = np.array(
            list(self.pow_timing_leading_zero_bits.keys()))
        self.max_timing = self.all_pow_timings_only[-1]

        self.all_bits_only = np.array(
            list(self.pow_timing_leading_zero_bits.values()))

        is_sorted = lambda a: np.all(a[:-1] <= a[1:])
        if is_sorted(self.all_pow_timings_only) is False:
            raise Exception(
                "The pow_timing_per_query array is required to be sorted. "
                "This ensures that we can do binary search on the timings.")

        self.model_from_queries_to_privacy = self._fit_linear_regression_from_queries_to_privacy() # model to predict the cost for a standard user.

        #self.model_from_cost_to_timing = self._fit_linear_regression_from_cost_to_timing()
        self.model_from_timing_to_bits = self._fit_linear_regression_from_timing_to_bits()

        # The model to map directly from the query cost to the number of leading
        # zero bits used in the PoW puzzle. Do not use this one - it gives poor
        # predictions. Use the above "double" mapping.
        self.model_from_cost_to_bits = self._fit_linear_regression_from_cost_to_bits()

        self.pow_leading_zero_bits_timing = {
            v: k for k, v in self.pow_timing_leading_zero_bits.items()}
        if dataset == "mnist":
            self.base = 1.0075
        elif dataset == "svhn":
            self.base = 1.0075
        elif dataset == "cifar10":
            self.base = 1.0075
        if self.type == "pknn":
            if dataset == "mnist":
                self.base = 1.18
            elif dataset == "svhn":
                self.base = 1.05

    def _get_new_timing(self, time: np.ndarray):
        y = time
        last_time = y[-1]

        # Use geometric series to compute PoW timing for privacy costs from
        # legitimate users.
        # The final y should be the PoW timings for the consecutive queries
        # so that the total execution time is around 2X longer for a legitimate
        # user. We assume each query batch has the same number (e.g., 64) of
        # samples.
        # a + ar + ar**2 + ar**3 + ... = Sum = a / (1 - r) = last_time
        # a = last_time * (1 - r)
        r = 0.5
        a = last_time * (1 - r)
        y = [a * r ** exp for exp in range(len(y))]
        y = np.array(y)
        # The timings were in the descending order but we need them in the
        # ascending order so flip y.
        y = np.flip(y)  # timing
        print('new timing for model: ', y, ' sum: ', np.sum(y), ' y: ',
              ",".join([str(x) for x in y]))
        return y

    def _get_privacy_cost_timing_for_legitimate_user(self):
        file = f'time_privacy_cost_{self.dataset}.csv'
        file2 = f'../pow/time_privacy_cost_{self.dataset}.csv'
        try:
            time_cost = genfromtxt(f'{file}', delimiter=',', skip_header=1)
        except:
            time_cost = genfromtxt(f'{file2}', delimiter=',', skip_header=1)
        # print('time_cost: ', time_cost)

        X = time_cost[:, 1].reshape(-1, 1)  # privacy cost
        y = time_cost[:, 0]  # timing

        new_y = y

        return X, new_y

    def _get_privacy_cost_queries_for_legitimate_user(self):
        if self.type == None or self.type == "":
            file = f'time_privacy_cost_{self.dataset}.csv'
            file2 = f'../pow/time_privacy_cost_{self.dataset}.csv'
        else:
            file = f'time_privacy_cost_{self.dataset}2pknn.csv'
        try:
            time_cost = genfromtxt(f'{file}', delimiter=',', skip_header=1)
        except:
            time_cost = genfromtxt(f'{file2}', delimiter=',', skip_header=1)
        # print('time_cost: ', time_cost)

        X = time_cost[:, 1].reshape(-1, 1)  # privacy cost
        y = time_cost[:, 2]  # queries

        new_y = y

        return X, new_y

    def _fit_linear_regression_from_cost_to_timing(self) -> BaseEstimator:
        """
        Take the timing and privacy cost for a legitimate user who sends queries
        that are in random order.

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        """
        X, y = self._get_privacy_cost_timing_for_legitimate_user()

        print('X,', to_csv([x[0] for x in X]))
        print('y,', to_csv(y))

        model = LinearRegression().fit(X, y)
        score = model.score(X, y)
        print('from cost to timing:')
        print('score: ', score)
        print('model coefficients: ', model.coef_)
        print('model intercept: ', model.intercept_)
        self.model_from_cost_to_timing = model

        # Small test.
        new_cost = 10
        predicted_timing = self.get_pow_timing_from_privacy_cost(
            privacy_cost=new_cost)
        print(f'predicted_timing for cost {new_cost}: ', predicted_timing)

        timings = []
        for x in X:
            # print('x: ', x)
            time = self.get_pow_timing_from_privacy_cost(x[0])
            timings.append(time)
        print('timings,', to_csv(timings))

        return model
    def _fit_linear_regression_from_queries_to_privacy(self) -> BaseEstimator:
        """
        Take the timing and privacy cost for a legitimate user who sends queries
        that are in random order. Fit a model to predict the privacy of a normal user
        given the timing.

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        """
        X, y = self._get_privacy_cost_queries_for_legitimate_user()
        # X is the privacy, y is the number of queries. This implements a reverse mapping

        print('X,', to_csv([x[0] for x in X]))
        print('y,', to_csv(y))

        y = np.array(y).reshape(-1, 1)
        X = np.array(X)

        model = LinearRegression().fit(y.reshape(len(y), 1), X)
        score = model.score(y, X)
        print('from queries to privacy:')
        print('score: ', score)
        print('model coefficients: ', model.coef_)
        print('model intercept: ', model.intercept_)
        self.model_from_queries_to_privacy = model

        # Small test.
        # new_time = 10
        # predicted_privacy = self._predict_standard_privacy(
        #     time=new_time)
        # print(f'predicted privacy for time {new_time}: ', predicted_privacy)

        # timings = []
        # for x in X:
        #     # print('x: ', x)
        #     time = self.get_pow_timing_from_privacy_cost(x[0])
        #     timings.append(time)
        # print('timings,', to_csv(timings))

        return model

    def _fit_linear_regression_from_timing_to_bits(self) -> BaseEstimator:
        """
        Take the timing and predict how many leading zero bits should be set for
        a challenge.

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        """
        X = self.all_pow_timings_only
        y = self.all_bits_only

        # Remove zero timing and zero bits
        X = X[1:]  # time
        y = y[1:]  # bits

        if self.use_exp:
            # The timing grows exponentially for a given number of bits so to make
            # this into a linear prediction we take the logarithm of the timing.
            X = np.log10(X)

        X = np.array(X).reshape(-1, 1)  # time
        y = np.array(y)  # bits

        model = LinearRegression().fit(X, y)
        score = model.score(X, y)
        print('from timing to bits:')
        print('score: ', score)
        print('model coefficients: ', model.coef_)
        print('model intercept: ', model.intercept_)

        # Small test.
        new_timing = 6000
        predicted_bits = model.predict(np.array([[np.log10(new_timing)]]))
        print(f'predicted bits for new timing {new_timing}: ', predicted_bits)

        return model

    def _get_bits_for_timing(self, timing):
        """

        Args:
            timing: the timing incurred by a puzzle

        Returns:
            how many bits should be set for the puzzle to get this timing

        """
        index = np.searchsorted(self.all_pow_timings_only, timing)
        if index == len(self.all_pow_timings_only):
            index = -1
        bits = self.all_bits_only[index]
        # pow_timing = self.all_pow_timings_only[index]
        # print('index: ', index, ' pow_timing: ', pow_timing, ' bits: ', bits)
        return bits

    def _fit_linear_regression_from_cost_to_bits(self) -> BaseEstimator:
        """
        Take the privacy cost for a legitimate user who is assumed to send
        in-distribution queries that are in random order and predict how many
        leading zero bits should be set for the puzzle.

        https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

        """
        X, y = self._get_privacy_cost_timing_for_legitimate_user()

        # map from y (timing) to z (number of bits)
        z = np.array([self._get_bits_for_timing(timing) for timing in y])

        # Exclude the values of bits <= 1.
        w = z[z > 1]
        X = X[z > 1]

        # Map from the privacy cost to the number of bits.
        model = LinearRegression().fit(X, w)
        score = model.score(X, w)
        print('from cost to bits:')
        print('score: ', score)

        # Small test.
        new_cost = 10
        predicted_bits = model.predict(np.array([[new_cost]]))
        print(f'predicted_bits for cost {new_cost}: ', predicted_bits)

        return model


    def _predict_pow_timing(self, diff: float) -> float:
        """
        Predict PoW timing for this difference and a legitimate user.

        Args:
            diff: the difference between the expected privacy cost for a standard user and the
            current cost.

        Returns:
            the expected timing for a legitimate user based on our prediction
            model

        """
        pow_timing = self.base ** diff # The base can be adjusted.
        # The PoW timing should be at least 1 sec.
        pow_timing = max(self.min_timing, pow_timing)
        # The PoW should not be longer than our current max timing.
        pow_timing = min(pow_timing, self.max_timing)
        return pow_timing

    def _predict_standard_privacy(self, queries: float) -> float:
        """
        Predict standard privacy cost for this time and a legitimate user.
        """
        pow_privacy = self.model_from_queries_to_privacy.predict(
            np.array([[queries]]))[0]
        return pow_privacy
    def _predict_leading_zero_bits_from_timing(self, timing: float) -> int:
        """
        Predict number of leading zero bits that should be assigned for the next
        challenge (PoW puzzle) based on the required timing.

        Args:
            timing: the timing requied for the next PoW.

        Returns:
            Number of leading zero bits to be set for the next challenge.
        """
        num_zero_bits = self.model_from_timing_to_bits.predict(
            np.array([[np.log10(timing)]]))[0]
        index = np.searchsorted(self.all_pow_timings_only, self.min_timing)
        if index == len(self.all_pow_timings_only):
            min_zero_bits = len(self.all_pow_timings_only)
        else:
            min_zero_bits = index

        num_zero_bits = max(num_zero_bits, min_zero_bits)
        return num_zero_bits

    def _predict_leading_zero_bits_from_cost(self, cost: float) -> float:
        """
        Predict the number of the leading zero bits for the PoW puzzle from the
        query cost.

        Args:
            cost: the query cost incurred by a user.

        Returns:
            Number of leading zero bits for the PoW puzzle.
        """
        nr_zero_bits = self.model_from_cost_to_bits.predict(
            np.array([[cost]]))[0]
        return nr_zero_bits

    def get_pow_timing_from_privacy_cost(self, privacy_cost: float, queries: float) -> float:
        """
        Get PoW timing.

        Args:
            privacy_cost: computed privacy cost up to now for the user.
            queries: current number of queries for the user.

        Returns:
            pow_timing: additional timing at this privacy cost
            (per batch of queries).

        """

        expected_privacy = self._predict_standard_privacy(queries=queries) # expected privacy cost of a normal user at this time
        diff = np.abs(privacy_cost - expected_privacy)  # absolute value of difference between actual cost and expected costs to use for the timing calculation
        # This timing is cumulative.
        expected_timing = self._predict_pow_timing(
             diff=diff)
        if queries == 0:
            expected_timing -= 1  # remove the extra 1 second at the beginning
        elif self.type == "pknn":  # data is in batches of 64 which for mnist takes less than 1 second which means that the time may exceed 2x for standard.
            if self.dataset == "mnist":
                expected_timing -= 0.75
        # We allow the PoW to take the same time as the expected timing.
        return expected_timing

    def get_leading_zero_bits_for_challenge_through_time(
            self, privacy_cost: float, queries:float) -> int:
        """
        The challenge should be for a query with the specified self.batch_size.

        Args:
            privacy_cost: the total (cumulative) privacy cost incurred by
                queries sent by a user

        Returns:
            Number of leading zero bits that should be used in the challenge
                (PoW puzzle) for the incurred privacy cost.

        """
        timing = self.get_pow_timing_from_privacy_cost(
            privacy_cost=privacy_cost, queries=queries)
        num_zero_bits = self._predict_leading_zero_bits_from_timing(
            timing=timing[0])
        return int(num_zero_bits)

    def get_leading_zero_bits_for_challenge_directly_from_cost(
            self, cost: float) -> int:
        """
        The challenge should be for a query with the specified self.batch_size.

        Args:
            cost: the total (cumulative) query cost incurred by a user.

        Returns:
            Number of leading zero bits that should be used in the challenge
                (PoW puzzle) for the incurred privacy cost.
        """
        num_zero_bits = self._predict_leading_zero_bits_from_cost(cost=cost)
        return int(num_zero_bits)

    def recompute_timings(
            self,
            timings: np.ndarray,
            privacy_costs: np.ndarray,
            queries_per_epoch: np.ndarray) -> np.ndarray:
        """
        For the raw timing of the attack, add the additional cost due to proof
        of work based on how many queries are run per epoch/batch at this
        timing and what privacy cost is incurred.

        Args:
            timings: cumulative execution time in sec for querying the ML API
                when no PoW is applied. This is based on the initial timing of
                the queries.
            privacy_costs: cumulative privacy cost incurred by the queries up to
                now.
            queries_per_epoch: queries send to the server per epoch / step, this
                is not a cumulative count.

        Returns:
            updated timings with added proof-of-work (PoW) depending on the
            privacy_cost in given epoch/step and number of queries per
            epoch/step

        """
        if len(timings) != len(privacy_costs):
            raise Exception('We have to have privacy cost for each timing.')
        # Cumulative timing for queries after applying PoW.
        pow_timings = np.zeros_like(timings)
        for idx, num_new_queries in enumerate(queries_per_epoch):
            # The privacy cost for new queries comes from the privacy cost
            # incurred by all previous queries.
            if idx == 0:
                privacy_cost = 0
                diff_timing = 0
                curtime = 0
                totalqueries=0
                previous_pow_timing = timings[0]
            else:
                privacy_cost = privacy_costs[idx]
                diff_timing = timings[idx] - timings[idx - 1]
                previous_pow_timing = pow_timings[idx - 1]
                totalqueries += (1000/16) * num_new_queries
            # additional delay per query because of PoW
            pow_timing = self.get_pow_timing_from_privacy_cost(
                privacy_cost=privacy_cost, queries=totalqueries)
            # additional delay for new queries
            # pow_delay = pow_timing * num_new_queries
            pow_delay = pow_timing
            new_queries_timing = diff_timing + pow_delay
            pow_timings[idx] = previous_pow_timing + new_queries_timing
        return pow_timings

    def recompute_timings_with_pow_bits(
            self,
            timings: np.ndarray,
            privacy_costs: np.ndarray) -> np.ndarray:
        """
        For the raw timing of the attack, add the additional cost due to proof
        of work based on privacy cost is incurred.

        Args:
            timings: cumulative execution time in sec for querying the ML API
                when no PoW is applied. This is based on the initial timing of
                the queries.
            privacy_costs: cumulative privacy cost incurred by the queries up to
                now.

        Returns:
            updated timings with added proof-of-work (PoW) depending on the
            privacy_cost. This PoW timing is also cumulative.

        """
        if len(timings) != len(privacy_costs):
            raise Exception('We have to have privacy cost for each timing.')
        # Cumulative timing for queries after applying PoW.
        pow_timings = np.zeros_like(timings)
        for idx, timing in enumerate(timings):
            # The privacy cost for new queries comes from the privacy cost
            # incurred by all previous queries.
            if idx == 0:
                privacy_cost = 0
                diff_timing = 0
                previous_pow_timing = timings[0]
            else:
                privacy_cost = privacy_costs[idx - 1]
                diff_timing = timings[idx] - timings[idx - 1]
                previous_pow_timing = pow_timings[idx - 1]
            # additional delay per query because of PoW
            # pow_timing = self.get_pow_timing_from_privacy_cost(
            #     privacy_cost=privacy_cost)
            bits = self.get_leading_zero_bits_for_challenge_directly_from_cost(
                cost=privacy_cost)
            pow_timing = self.pow_leading_zero_bits_timing[bits]

            new_queries_timing = diff_timing + pow_timing
            pow_timings[idx] = previous_pow_timing + new_queries_timing
        return pow_timings


if __name__ == "__main__":
    # Small test.
    pow = PoW(dataset='mnist')
    queries = 20000
    privacy_cost = 5150
    print("Predicted privacy cost", pow._predict_standard_privacy(queries))
    print("POW Time (to be added)", pow.get_pow_timing_from_privacy_cost(
        privacy_cost=privacy_cost, queries=queries))
    # query_cost = 5000
    leading_bit_zeros1 = pow.get_leading_zero_bits_for_challenge_through_time(
        privacy_cost=privacy_cost, queries=queries)
    print('leading_bit_zeros obtained through time: ', leading_bit_zeros1)
    #
    # leading_bit_zeros2 = pow.get_leading_zero_bits_for_challenge_directly_from_cost(
    #     cost=query_cost)
    # print('leading_bit_zeros obtained directly from the query cost: ',
    #       leading_bit_zeros2)