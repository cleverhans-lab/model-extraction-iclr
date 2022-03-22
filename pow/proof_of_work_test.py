import numpy as np
import pandas as pd
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path().absolute().parent))
from pow.proof_of_work import PoW


def get_time_privacy(data, mode='standard'):
    time = data[data['mode'] == mode]['time'].to_numpy()
    privacy = data[data['mode'] == mode]['privacy'].to_numpy()
    queries = data[data['mode'] == mode]['diff queries'].to_numpy()
    return time, privacy, queries


def to_csv(array: np.ndarray):
    return ",".join([str(x) for x in array])


def main():
    name = 'pow_standard_copycat'
    data = pd.read_csv(f"{name}.csv")

    standard_time, standard_privacy, standard_queries = get_time_privacy(data=data,
                                                       mode='standard')
    copycat_time, copycat_privacy, copycat_queries = get_time_privacy(data=data, mode='copycat')

    pow = PoW(dataset='cifar10', batch_size=16)

    standard_pow_time = pow.recompute_timings(timings=standard_time,
                                                   privacy_costs=standard_privacy, queries_per_epoch=standard_queries)
    print('standard time,', to_csv(standard_time))
    print('standard pow time,', to_csv(standard_pow_time))

    copycat_pow_time = pow.recompute_timings(timings=copycat_time,
                                                  privacy_costs=copycat_privacy, queries_per_epoch=copycat_queries)
    print('copycat_time,', to_csv(copycat_time))
    print('copytcat_pow_time,', to_csv(copycat_pow_time))


if __name__ == "__main__":
    main()
