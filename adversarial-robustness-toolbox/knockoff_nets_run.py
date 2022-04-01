from knockoff_nets import TestKnockoffNets
import argparse

parser = argparse.ArgumentParser(description='Adaptive Knockoff')
parser.add_argument('--dataset', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'svhn'])
parser.add_argument('--num_queries', default=4000, type=int,
                    help='Number of queries to steal the model.')


if __name__ == "__main__":
    args = parser.parse_args()
    if args.dataset == 'cifar10':
        train = False
    elif args.dataset == 'mnist':
        train = True
    elif args.dataset == 'svhn':
        train = False
    else:
        raise Exception(f"Unknown dataset: {args.dataset}")
    if train:
        load_init = False
    else:
        load_init = True

    knockoff = TestKnockoffNets(
        train=train,
        random=False,
        adaptive=True,
        dataset=args.dataset,
        load_init=load_init,
        NB_STOLEN = args.num_queries)
    knockoff.runknockoff()
