from __future__ import print_function
import argparse, json, ipdb
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader, Subset, ConcatDataset
import network
from dataloader import get_dataloader
import os, random
import numpy as np
import torchvision
from pprint import pprint
from time import time
import scipy.stats
from private_knn import PrivateKnn

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path().absolute().parent.parent))

from models.ensemble_model import EnsembleModel
from models.load_models import load_private_models
from models.load_models import load_victim_model
from models.private_model import get_private_model_by_id
import analysis

import socket
import pickle
from time import sleep
HOST = '127.0.0.1'  # The server's hostname or IP address
PORT = 65432  # The port used by the server
server_address = (HOST, PORT)

from approximate_gradients import *

import torchvision.models as models
from my_utils import *

print("torch version", torch.__version__)

cuda = torch.cuda.is_available()

def myprint(a):
    """Log the print statements"""
    global file
    print(a);
    file.write(a);
    file.write("\n");
    file.flush()


def student_loss(args, s_logit, t_logit, return_t_logits=False):
    """Kl/ L1 Loss for student"""
    print_logits = False
    if args.loss == "l1":
        loss_fn = F.l1_loss
        loss = loss_fn(s_logit, t_logit.detach())
    elif args.loss == "kl":
        loss_fn = F.kl_div
        s_logit = F.log_softmax(s_logit, dim=1)
        t_logit = F.softmax(t_logit, dim=1)
        loss = loss_fn(s_logit, t_logit.detach(), reduction="batchmean")
    else:
        raise ValueError(args.loss)

    if return_t_logits:
        return loss, t_logit.detach()
    else:
        return loss


def generator_loss(args, s_logit, t_logit, z=None, z_logit=None,
                   reduction="mean"):
    assert 0

    loss = - F.l1_loss(s_logit, t_logit, reduction=reduction)

    return loss


def train(args, teacher, student, generator, device, optimizer, epoch):
    """Main Loop for one epoch of Training Generator and Student"""
    global file
    if args.usenetwork == "yes":
        usenetwork = True
    else:
        usenetwork = False
    timecost = 0
    timequery = 0
    teacher.eval()
    student.train()
    trainloader, _ = get_dataloader(args)
    pate_knn = PateKNN(model=teacher, trainloader=trainloader,
                       args=args)
    if args.dataset == "cifar10":
        model_path = f"../../private-models/{args.dataset}/ResNet18/50-models"
        args.architectures = ["ResNet18"]
    elif args.dataset == "mnist":
        model_path = f"../../private-models/{args.dataset}/MnistNetPate/250-models"
        args.architectures = ["MnistNetPate"]
    else:
        model_path = f"../../private-models/{args.dataset}/ResNet10/250-models"
        args.architectures = ["ResNet10"]
    args.xray_datasets = ['AP']
    args.cuda = torch.cuda.is_available()
    private_models = load_private_models(args=args,
                                         model_path=model_path)
    victim_model = EnsembleModel(model_id=-1, args=args,
                                 private_models=private_models)
    optimizer_S, optimizer_G = optimizer

    gradients = []
    entropy_cost = 0  # entropy cost for this epoch
    gap_cost = 0  # gap cost for the current epoch
    tlogits = None
    tqueries = None
    for i in range(args.epoch_itrs):
        """Repeat epoch_itrs times per epoch"""
        for _ in range(args.g_iter):
            # Sample Random Noise
            z = torch.randn((args.batch_size, args.nz)).to(device)
            optimizer_G.zero_grad()
            generator.train()
            # Get fake image from generator
            fake = generator(z,
                             pre_x=args.approx_grad)

            approx_grad_wrt_x, loss_G, preds, quers, timequery1 = estimate_gradient_objective(args,
                                                                    teacher,
                                                                    student,
                                                                    fake,
                                                                    epsilon=args.grad_epsilon,
                                                                    m=args.grad_m,
                                                                    num_classes=args.num_classes,
                                                                    device=device,
                                                                    pre_x=True)

            fake.backward(approx_grad_wrt_x)

            optimizer_G.step()
            start_time = time()
            entropy_scores = computeentropy(preds)
            entropy_cost += entropy_scores.sum()
            gap_scores = computegap(preds)
            gap_cost += gap_scores.sum()
            end_time = time()
            timecost += end_time - start_time
            timequery += timequery1
            if tlogits == None:
                tlogits = preds
                tqueries = quers
            else:
                tlogits = torch.cat((tlogits, preds), dim=0)
                tqueries = torch.cat((tqueries, quers), dim=0)

            if i == 0 and args.rec_grad_norm:
                x_true_grad = measure_true_grad_norm(args, fake)
        for _ in range(args.d_iter):
            z = torch.randn((args.batch_size, args.nz)).to(device)
            fake = generator(z).detach()
            optimizer_S.zero_grad()
            if args.dataset == "mnist" or args.dataset == "fashion-mnist":
                with torch.no_grad():
                    temp2 = []
                    for blah in range(fake.size(0)):
                        temp = fake[blah]
                        temp = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Grayscale(num_output_channels=1),
                            transforms.ToTensor(), ])(temp)
                        temp2.append(temp.reshape(1, 1, 32,
                                                  32))
                    fake2 = torch.cat(temp2)
                    fake2 = fake2.to(device)
                    timequery21 = time()
                    if usenetwork:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.connect(server_address)
                        try:
                            datastr = pickle.dumps(fake2)
                            sock.sendall(datastr)
                            sleep(0.1)
                            str = "done"
                            sock.sendall(str.encode())

                            ### POW Challenge
                            challenge = sock.recv(4096)
                            challenge = pickle.loads(challenge)
                            pos = challenge.find(":")
                            pos2 = challenge[pos+1:].find(":")
                            bits = challenge[pos+1:pos+pos2+1]
                            bits = int(bits)
                            xtype = 'bin'
                            stamp = mint_iteractive(challenge=challenge, bits=bits, xtype=xtype)
                            datastamp = pickle.dumps(stamp)
                            sock.sendall(datastamp)
                            output = sock.recv(4096)
                            t_logit = pickle.loads(output)

                            # For larger batch sizes.
                            # data = []
                            # while True:
                            #     packet = sock.recv(4096)
                            #     if packet == b'donesend':
                            #         break
                            #     # print(packet)
                            #     if not packet or packet == b'done': break
                            #     data.append(packet)
                            #t_logit = pickle.loads(b"".join(data))

                            sleep(0.1)
                            str = "doneiter"
                            sock.sendall(str.encode())
                        finally:
                            sock.close()
                    else:
                        t_logit = teacher(fake2)
                        #print("size recevied", t_logit.size())
                    timequery22 = time()
                    timequery2 = timequery22 - timequery21
                    if tlogits == None:
                        tlogits = t_logit
                        tqueries = fake2
                    else:
                        tlogits = torch.cat((tlogits, t_logit), dim=0)
                        tqueries = torch.cat((tqueries, fake2), dim=0)
                    # Entropy and Gap Calculation
                    start_time = time()
                    entropy_scores = computeentropy(t_logit)
                    gap_scores = computegap(t_logit)
                    entropy_cost += entropy_scores.sum()
                    gap_cost += gap_scores.sum()
                    end_time = time()
                    timecost += end_time - start_time
                    timequery += timequery2
            else:
                with torch.no_grad():
                    timequery21 = time()
                    t_logit = teacher(fake)
                    timequery22 = time()
                    timequery2 = timequery22 - timequery21
                    if tlogits == None:
                        tlogits = t_logit
                        tqueries = fake
                    else:
                        tlogits = torch.cat((tlogits, t_logit), dim=0)
                        tqueries = torch.cat((tqueries, fake), dim=0)
                    # Entropy and Gap Calculation
                    start_time = time()
                    entropy_scores = computeentropy(t_logit)
                    gap_scores = computegap(t_logit)
                    entropy_cost += entropy_scores.sum()
                    gap_cost += gap_scores.sum()
                    end_time = time()
                    timecost += end_time - start_time
                    timequery += timequery2

            # Correction for the fake logits
            if args.loss == "l1" and args.no_logits:
                t_logit = F.log_softmax(t_logit, dim=1).detach()
                if args.logit_correction == 'min':
                    t_logit -= t_logit.min(dim=1).values.view(-1, 1).detach()
                elif args.logit_correction == 'mean':
                    t_logit -= t_logit.mean(dim=1).view(-1, 1).detach()

            s_logit = student(fake)

            loss_S = student_loss(args, s_logit, t_logit)
            loss_S.backward()
            optimizer_S.step()
        # Log Results
        if i % args.log_interval == 0:
            myprint(
                f'Train Epoch: {epoch} [{i}/{args.epoch_itrs} ({100 * float(i) / float(args.epoch_itrs):.0f}%)]\tG_Loss: {loss_G.item():.6f} S_loss: {loss_S.item():.6f}')

            if i == 0:
                with open(args.log_dir + "/loss.csv", "a") as f:
                    f.write("%d,%f,%f\n" % (epoch, loss_G, loss_S))

            if args.rec_grad_norm and i == 0:

                G_grad_norm, S_grad_norm = compute_grad_norms(generator,
                                                              student)
                if i == 0:
                    with open(args.log_dir + "/norm_grad.csv", "a") as f:
                        f.write("%d,%f,%f,%f\n" % (
                            epoch, G_grad_norm, S_grad_norm, x_true_grad))

        # update query budget
        args.query_budget -= args.cost_per_iteration

        if args.query_budget < args.cost_per_iteration:
            break  # return
    # PATE cost
    start_time = time()
    if args.dataset == "mnist":
        trans = transforms.Compose([transforms.Resize(28)])
        tqueries = trans(tqueries)
        tdataset = [(a, 0) for a in tqueries]
    else:
        tdataset = [(a, 0) for a in tqueries]
    adaptive_loader = DataLoader(
            tdataset,
            batch_size=1,
            shuffle=False)
    votes_victim = victim_model.inference(adaptive_loader, args)
    datalength = len(votes_victim)
    pate_cost = 0
    for i in range(datalength):
        curvote = votes_victim[i][np.newaxis, :]
        max_num_query, dp_eps, partition, answered, order_opt = analysis.analyze_multiclass_confident_gnmax(
            votes=curvote,
            threshold=0,
            sigma_threshold=0,
            sigma_gnmax=args.sigma_gnmax,
            budget=np.inf,
            file=None,
            delta=args.delta,
            show_dp_budget=args.show_dp_budget,
            args=args
        )
        print(f'dp_eps for vote {i}: {dp_eps[0]}')
        pate_cost += dp_eps[0]
    print("pate cost", pate_cost)
    end_time = time()
    timecost += end_time - start_time
    with open(args.log_dir + "/entropy.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, entropy_cost))
    with open(args.log_dir + "/gap.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, gap_cost))
    with open(args.log_dir + "/pate.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, pate_cost))
    return timecost, timequery


def test(args, student=None, generator=None, device="cuda", test_loader=None,
         epoch=0):
    global file
    student.eval()
    generator.eval()

    test_loss = 0
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if args.dataset == 'mnist' or args.dataset == "fashion-mnist":
                data2 = data.repeat(1, 3, 1, 1)
                output = student(data2)
            else:
                output = student(data)

            test_loss += F.cross_entropy(output, target,
                                         reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    myprint(
        '\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            accuracy))
    with open(args.log_dir + "/accuracy.csv", "a") as f:
        f.write("%d,%f\n" % (epoch, accuracy))
    acc = correct / len(test_loader.dataset)
    return acc


def compute_grad_norms(generator, student):
    G_grad = []
    for n, p in generator.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            G_grad.append(p.grad.norm().to("cpu"))

    S_grad = []
    for n, p in student.named_parameters():
        if "weight" in n:
            # print('===========\ngradient{}\n----------\n{}'.format(n, p.grad.norm().to("cpu")))
            S_grad.append(p.grad.norm().to("cpu"))
    return np.mean(G_grad), np.mean(S_grad)


def computeentropy(t_logits):
    num_classes = 10
    entropy = []
    prob = F.softmax(t_logits, dim=1).cpu().numpy()
    entropy.append(scipy.stats.entropy(prob, axis=1))
    entropy = np.concatenate(entropy, axis=0)
    entropy_max = np.log(10)
    utility = entropy / entropy_max
    return utility


def computegap(t_logits):
    gap = []
    sorted_output = t_logits.sort(dim=-1, descending=True)[0]
    prob = F.softmax(sorted_output[:, :2], dim=1).cpu().numpy()
    gap.append(prob[:, 0] - prob[:, 1])
    gap = np.concatenate(gap, axis=0)
    utility = 1 - gap
    return utility

def get_votes_for_pate_knn(model, t_logits, train_represent,
                           train_labels, args = None):
    """
    :param model: the model to be used
    :param unlabeled_loader: data points to be labeled - for which we compute
        the score
    :param train_represent: last layer representation for the teachers
    :param train_labels: labels for the teachers
    :param args: the program parameters
    :return: votes for each data point
    """

    # num_teachers: number of k nearest neighbors acting as teachers
    num_teachers = 300

    with torch.no_grad():
        # Privacy cost as a proxy for utility.
        votes = []
        predictions = []
        outputs = F.log_softmax(t_logits, dim=-1)
        outputs = outputs.cpu().numpy()
        predictions.append(np.argmax(outputs, axis=-1))
        for output in outputs:
            dis = np.linalg.norm(train_represent - output, axis=-1)
            k_index = np.argpartition(dis, kth=num_teachers)[:num_teachers]
            teachers_preds = np.array(train_labels[k_index], dtype=np.int32)
            label_count = np.bincount(
                teachers_preds, minlength=10)
            votes.append(label_count)
    votes = np.stack(votes)
    return votes


class PateKNN:
    """
    Compute the privacy cost.
    """

    def __init__(self, model, trainloader, args):
        """
        Args:
            model: the victim model.
            trainloader: the data loader for the training data.
            args: the program parameters.
        """
        self.model = model
        self.args = args

        # Extract the last layer representation of the training points and their
        # ground-truth labels.
        train_represent = []
        train_labels = []
        with torch.no_grad():
            for batch_id, (data, target) in enumerate(trainloader):
                if cuda:
                    data = data.cuda()
                outputs = model(data)
                outputs = F.log_softmax(outputs, dim=-1)
                outputs = outputs.cpu().numpy()
                train_represent.append(outputs)
                train_labels.append(target.cpu().numpy())
        self.train_represent = np.concatenate(train_represent, axis=0)
        self.train_labels = np.concatenate(train_labels, axis=0)

        self.private_knn = PrivateKnn(
            delta=1e-5, sigma_gnmax=28, #28
            apply_data_independent_bound=False)

    def compute_privacy_cost(self, t_logits):
        """
        Args:
            unlabeled_loader: data loader for new queries.
        Returns:
            The total privacy cost incurred by all the queries seen so far.
        """
        votes = get_votes_for_pate_knn(
            model=self.model, t_logits=t_logits, train_labels=self.train_labels,
            train_represent=self.train_represent, args=self.args
        )

        dp_eps = self.private_knn.add_privacy_cost(votes=votes)

        return dp_eps


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='DFAD CIFAR')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 256)')
    parser.add_argument('--query_budget', type=float, default=20, metavar='N',
                        help='Query budget for the extraction attack in millions (default: 20M)')
    parser.add_argument('--epoch_itrs', type=int, default=50)
    parser.add_argument('--g_iter', type=int, default=1,
                        help="Number of generator iterations per epoch_iter")
    parser.add_argument('--d_iter', type=int, default=5,
                        help="Number of discriminator iterations per epoch_iter")

    parser.add_argument('--lr_S', type=float, default=0.1, metavar='LR',
                        help='Student learning rate (default: 0.1)')
    parser.add_argument('--lr_G', type=float, default=1e-4,
                        help='Generator learning rate (default: 0.1)')
    parser.add_argument('--nz', type=int, default=256,
                        help="Size of random noise input to generator")

    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--loss', type=str, default='l1',
                        choices=['l1', 'kl'], )
    parser.add_argument('--scheduler', type=str, default='multistep',
                        choices=['multistep', 'cosine', "none"], )
    parser.add_argument('--steps', nargs='+', default=[0.1, 0.3, 0.5],
                        type=float,
                        help="Percentage epochs at which to take next step")
    parser.add_argument('--scale', type=float, default=3e-1,
                        help="Fractional decrease in lr")

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['svhn', 'cifar10', 'mnist', 'fashion-mnist'],
                        help='dataset name (default: cifar10)')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--num_models', type=int, default=50)
    parser.add_argument('--sigma_gnmax', type=int, default=10)
    parser.add_argument('--delta', type=float, default=1e-5)
    parser.add_argument('--model', type=str, default='lenet5',
                        choices=classifiers,
                        help='Target model name (default: resnet34_8x)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=random.randint(0, 100000),
                        metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--ckpt', type=str,
                        default='checkpoint/teacher/mnist-lenet5.pt')

    parser.add_argument('--student_load_path', type=str, default=None)
    parser.add_argument('--model_id', type=str, default="debug")

    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_dir', type=str, default="results")

    parser.add_argument('--usenetwork', type=str, default="no")

    # Gradient approximation parameters
    parser.add_argument('--approx_grad', type=int, default=1,
                        help='Always set to 1')
    parser.add_argument('--grad_m', type=int, default=1,
                        help='Number of steps to approximate the gradients')
    parser.add_argument('--grad_epsilon', type=float, default=1e-3)

    parser.add_argument('--forward_differences', type=int, default=1,
                        help='Always set to 1')

    # Eigenvalues computation parameters
    parser.add_argument('--no_logits', type=int, default=1)
    parser.add_argument('--logit_correction', type=str, default='mean',
                        choices=['none', 'mean'])

    parser.add_argument('--rec_grad_norm', type=int, default=1)

    parser.add_argument('--MAZE', type=int, default=0)

    parser.add_argument('--store_checkpoints', type=int, default=0)

    parser.add_argument('--student_model', type=str, default='resnet18_8x',
                        help='Student model architecture (default: resnet18_8x)')
    parser.add_argument(
        '--vote_type', type=str,
        default='discrete',
        help='The type of votes. Discrete - each vote is a single number 0 or 1,'
             'or probability - the probability of a label being one.'
    )
    parser.add_argument(
        '--show_dp_budget', type=str,
        default='disable',
        # default='disable',
        help='Apply or disable showing the current privacy budget.'
    )

    args = parser.parse_args()

    args.query_budget *= 10 ** 6
    args.query_budget = int(args.query_budget)
    if args.MAZE:
        print("\n" * 2)
        print("#### /!\ OVERWRITING ALL PARAMETERS FOR MAZE REPLCIATION ####")
        print("\n" * 2)
        args.scheduer = "cosine"
        args.loss = "kl"
        args.batch_size = 128
        args.g_iter = 1
        args.d_iter = 5
        args.grad_m = 10
        args.lr_G = 1e-4
        args.lr_S = 1e-1

    if args.student_model not in classifiers:
        if "wrn" not in args.student_model:
            raise ValueError("Unknown model")

    pprint(args, width=80)
    print(args.log_dir)
    os.makedirs(args.log_dir, exist_ok=True)

    if args.store_checkpoints:
        os.makedirs(args.log_dir + "/checkpoint", exist_ok=True)

    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    with open(args.log_dir + "/loss.csv", "w") as f:
        f.write("epoch,loss_G,loss_S\n")

    with open(args.log_dir + "/accuracy.csv", "w") as f:
        f.write("epoch,accuracy\n")

    with open(args.log_dir + "/entropy.csv", "w") as f:
        f.write("epoch,entropy\n")

    with open(args.log_dir + "/gap.csv", "w") as f:
        f.write("epoch,gap\n")
    with open(args.log_dir + "/pate.csv", "w") as f:
        f.write("epoch,pate\n")
    with open(args.log_dir + "/time.csv", "w") as f:
        f.write("queries,time\n")
    with open(args.log_dir + "/timequery.csv", "w") as f:
        f.write("queries,time\n")
    if args.rec_grad_norm:
        with open(args.log_dir + "/norm_grad.csv", "w") as f:
            f.write("epoch,G_grad_norm,S_grad_norm,grad_wrt_X\n")

    with open("latest_experiments.txt", "a") as f:
        f.write(args.log_dir + "\n")
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Preparing checkpoints for the best Student
    global file
    model_dir = f"checkpoint/student_{args.model_id}";
    args.model_dir = model_dir
    if (not os.path.exists(model_dir)):
        os.makedirs(model_dir)
    with open(f"{model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    file = open(f"{args.model_dir}/logs.txt", "w")

    print(args)

    args.device = device

    # Eigen values and vectors of the covariance matrix
    _, test_loader = get_dataloader(args)  # mnist test set.

    args.normalization_coefs = None
    args.G_activation = torch.tanh

    num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist', 'fashion-mnist'] else 100
    args.num_classes = num_classes

    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)
        if args.dataset == 'svhn':
            print("Loading SVHN TEACHER")
            args.ckpt = 'checkpoint/teacher/svhn-resnet34_8x.pt'
        elif args.dataset == 'cifar10':
            print("Loading cifar10 TEACHER")
            args.ckpt = 'checkpoint/teacher/cifar10-resnet34_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=device))
    elif args.model == 'lenet5':
        teacher = network.lenet.LeNet5()
        if args.dataset == 'mnist':
            print("Loading mnist TEACHER")
            args.ckpt = 'checkpoint/teacher/mnist-lenet5.pt'
        elif args.dataset == "fashion-mnist":
            print("Loading Fashion MNIST TEACHER")
            args.ckpt = 'checkpoint/teacher/fashion-mnist-lenet5.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=device))


    else:
        teacher = get_classifier(args.model, pretrained=True,
                                 num_classes=args.num_classes)

    teacher.eval()
    teacher = teacher.to(device)
    myprint("Teacher restored from %s" % (args.ckpt))
    print(f"\n\t\tTraining with {args.model} as a Target\n")
    correct = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = teacher(data)
            pred = output.argmax(dim=1,
                                 keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(correct,
                                                                     len(test_loader.dataset),
                                                                     accuracy))

    student = get_classifier(args.student_model, pretrained=False,
                             num_classes=args.num_classes)

    generator = network.gan.GeneratorA(nz=args.nz, nc=3, img_size=32,
                                       activation=args.G_activation)

    student = student.to(device)
    generator = generator.to(device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    if args.student_load_path:
        # "checkpoint/student_no-grad/cifar10-resnet34_8x.pt"
        student.load_state_dict(torch.load(args.student_load_path))
        myprint("Student initialized from %s" % (args.student_load_path))
        acc = test(args, student=student, generator=generator, device=device,
                   test_loader=test_loader)

    ## Compute the number of epochs with the given query budget:
    args.cost_per_iteration = args.batch_size * (args.g_iter * (
            args.grad_m + 1) + args.d_iter)  # = 256 * (1 * 2 + 5) = 1792

    number_epochs = args.query_budget // (
            args.cost_per_iteration * args.epoch_itrs) + 1  # 1 epoch corresponds to cost_per_iteration * epoch_itrs = 1792* 50 = 89600 queries.

    queryepoch = args.cost_per_iteration * args.epoch_itrs  # how many queries 1 epoch corresponds to

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)
    print("Number of queries per epoch", queryepoch)

    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S,
                            weight_decay=args.weight_decay, momentum=0.9)

    if args.MAZE:
        optimizer_G = optim.SGD(generator.parameters(), lr=args.lr_G,
                                weight_decay=args.weight_decay, momentum=0.9)
    else:
        optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: ", steps)
    print()

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps,
                                                     args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps,
                                                     args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S,
                                                           number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G,
                                                           number_epochs)

    best_acc = 0
    acc_list = []

    for epoch in range(1, number_epochs + 1):
        # Train
        start = time()
        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()
        timecost, timequery = train(args, teacher=teacher, student=student, generator=generator,
              device=device, optimizer=[optimizer_S, optimizer_G], epoch=epoch)
        end = time()
        with open(args.log_dir + "/time.csv", "a") as f:
            f.write("%d,%f\n" % (
            epoch, end - start - timecost))
        with open(args.log_dir + "/timequery.csv", "a") as f:
            f.write("%d,%f\n" % (
                epoch, timequery))

        # Test
        acc = test(args, student=student, generator=generator, device=device,
                   test_loader=test_loader, epoch=epoch)
        acc_list.append(acc)
        if acc > best_acc:
            best_acc = acc
            name = 'resnet34_8x'
            torch.save(student.state_dict(),
                       f"checkpoint/student_{args.model_id}/{args.dataset}-{name}.pt")
            torch.save(generator.state_dict(),
                       f"checkpoint/student_{args.model_id}/{args.dataset}-{name}-generator.pt")
        # vp.add_scalar('Acc', epoch, acc)
        if args.store_checkpoints:
            torch.save(student.state_dict(),
                       args.log_dir + f"/checkpoint/student.pt")
            torch.save(generator.state_dict(),
                       args.log_dir + f"/checkpoint/generator.pt")
    myprint("Best Acc=%.6f" % best_acc)

    with open(args.log_dir + "/Max_accuracy = %f" % best_acc, "w") as f:
        f.write(" ")

    import csv
    os.makedirs('log', exist_ok=True)
    with open('log/DFAD-%s.csv' % (args.dataset), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(acc_list)


if __name__ == '__main__':
    main()
