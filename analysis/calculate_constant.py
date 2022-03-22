import argparse
import pandas as pd
import numpy as np
import itertools

parser = argparse.ArgumentParser()
parser.add_argument('--num_labels', nargs='+', default=[2, 8, 18, 20], help='Number of labels to consider for calculations of C')
parser.add_argument('--epsilons', nargs='+', default=[20], help='Epsilon values to consider for calculations of C')
parser.add_argument('--deltas', nargs='+', default=[1e-6], help='Delta values to consider for calculations of C, these will be clipped if < exp(-k/log(k)**8)')
parser.add_argument('--num_models', nargs='+', default=[50, 100, 150, 200], help='Number of models to consider for calculations of C')
parser.add_argument('--linf_bounds', nargs='+', default=[1e-2, 1e-1, 1, 1e1], help='L_inf error bounds to consider for calculations of C')

args = parser.parse_args()

params = list(itertools.product(args.num_labels, args.epsilons, args.deltas, args.num_models, args.linf_bounds))
df = pd.DataFrame(data=params, columns=['k', 'eps', 'delta', 'n', 'linf'])
df.delta = df.delta.clip(lower=np.exp(-df.k/np.log(df.k)**8))
df['C'] = (df.linf*df.eps*df.n)/np.sqrt(df.k*np.log(1/df.delta))

idx_min = np.argmin(df.C.values)
idx_max = np.argmax(df.C.values)

print('Minimum C value: {}, for parameters: number of labels = {}, eps = {}, delta = {}, number of  models = {} and L_inf error bound = {}'.format(df.iloc[idx_min]['C'], df.iloc[idx_min]['k'], df.iloc[idx_min]['eps'], df.iloc[idx_min]['delta'], df.iloc[idx_min]['n'], df.iloc[idx_min]['linf']))

print('Maximum C value {} for parameters: number of labels = {}, eps = {}, delta = {}, number of models = {} and L_inf error bound = {}'.format(df.iloc[idx_max]['C'], df.iloc[idx_max]['k'], df.iloc[idx_max]['eps'], df.iloc[idx_max]['delta'], df.iloc[idx_max]['n'], df.iloc[idx_max]['linf']))

df.to_csv('calculated_universal_constants.csv', sep=',', header=True, index=False)
