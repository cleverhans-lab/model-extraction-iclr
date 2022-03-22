import socket
HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432  # Port to listen on (non-privileged ports are > 1023)

import torch
import numpy as np
import random
import torchvision
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
cuda = torch.cuda.is_available()
from analysis.private_knn import PrivateKnn
import analysis
from pow.hashcash import mint_iteractive, generate_challenge, check, _to_binary
from pow.proof_of_work import PoW
import dfmenetwork
import scipy
import scipy.stats
import math
import pickle
import time
from models.ensemble_model import EnsembleModel
from models.load_models import load_private_models
from models.load_models import load_victim_model
from models.private_model import get_private_model_by_id
from parameters import get_parameters
import argparse
import os


parser = argparse.ArgumentParser(description='Server Setup')
parser.add_argument('--dataset', default='mnist', type=str)
parser.add_argument('--mode', default='other', type=str, help="select type of attack being used (dfme or other)")
parser.add_argument('--path', default=f"/ssd003/home/{os.getenv('USER')}/data", type=str, help="path to datasets")
args = parser.parse_args()
args2 = get_parameters() # original parameters

DAY1 = 60 * 60 * 24  # Seconds in a day

print("cuda available", cuda)
dataset = args.dataset
mode = args.mode
args = None
# Initialization of PATE ensemble:
print(f"Using {dataset} dataset")
if dataset == "cifar10":
    args2.dataset = "cifar10"
    args2.begin_id = 0
    args2.end_id = 50
    args2.num_models =50
    args2.architecture = "ResNet18"
    args2.architectures = ["ResNet18"]
    args2.class_type = "multiclass"
    args2.target_model = "pate"
    args2.sigma_gnmax = 2
    args2.delta = 1e-5
    args2.private_model_path = "private-models/cifar10/ResNet18/50-models"
    args2.cuda = torch.cuda.is_available()
    args2.num_classes = 10
elif dataset == "mnist":
    args2.dataset = "mnist"
    args2.begin_id = 0
    args2.end_id = 250
    args2.num_models = 250
    args2.architecture = "MnistNetPate"
    args2.class_type = "multiclass"
    args2.target_model = "pate"
    args2.sigma_gnmax = 10
    args2.delta = 1e-5
    args2.private_model_path = "private-models/mnist/MnistNetPate/250-models"
    args2.cuda = torch.cuda.is_available()
    args2.num_classes = 10
else:
    args2.dataset = "svhn"
    args2.begin_id = 0
    args2.end_id = 250
    args2.num_models = 250
    args2.architecture = "ResNet10"
    args2.architectures = ["ResNet10"]
    args2.class_type = "multiclass"
    args2.target_model = "pate"
    args2.sigma_gnmax = 10
    args2.delta = 1e-6
    args2.private_model_path = "private-models/svhn/ResNet10/250-models"
    args2.cuda = torch.cuda.is_available()
    args2.num_classes = 10

private_models = load_private_models(args=args2,
                                             model_path=args2.private_model_path)
victim_model = EnsembleModel(model_id=-1, args=args2,
                                     private_models=private_models)
if dataset == "cifar10":
    victim = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
    ckpt = 'dfmodels/teacher/cifar10-resnet34_8x.pt'
elif dataset == "svhn":
    victim = dfmenetwork.resnet_8x.ResNet34_8x(num_classes=10)
    ckpt = 'dfmodels/teacher/svhn-resnet34_8x.pt'
else:
    def load_private_model():
        from architectures.mnist_net_pate import MnistNetPate
        filepath = "private-models/mnist/MnistNetPate/1-models/checkpoint-model(1).pth.tar"
        if os.path.isfile(filepath):
            model = MnistNetPate(name='model({:d})'.format(0 + 1), args = args)
            checkpoint = torch.load(filepath)
            model.load_state_dict(checkpoint['state_dict'])
            if cuda:
                model.cuda()
            if 'label_weights' in checkpoint and args.label_reweight is 'apply':
                model.label_weights = checkpoint['label_weights']
            model.eval()
            return model
        else:
            raise Exception(
                f"Checkpoint file {filepath} does not exist, please generate it via "
                f"train_private_models(args)!")
    if mode == "dfme":
        victim = dfmenetwork.lenet.LeNet5()
        ckpt = 'dfmodels/teacher/mnist-lenet5.pt'
    else:
        victim = load_private_model()
if dataset != "mnist" or mode == "dfme":
    if cuda:
        victim.load_state_dict(torch.load(ckpt))
        victim = victim.cuda()
    else:
        victim.load_state_dict(torch.load(ckpt), map_location=torch.device('cpu'))
victim.eval()
print("Done loading victim")
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((HOST, PORT))
print("Started server")
#s.listen(5)
i = 0
privacy_cost = 0
num_queries = 0 # cumulative query count
pow = PoW(dataset=dataset)
while i<=1000:
    if i % 500 == 0:
        s.listen(1)
        conn, addr = s.accept()
        print('Connected by', addr)
    i += 1
    data = []
    while True:
        packet = conn.recv(4096)
        if packet == b'doneiter':
            i = 0
            break
        #print(packet)
        if not packet or packet == b'done':
            break
        data.append(packet)
        #print("rec pack")
    #print("Done")
    if i >= 1:
        data = pickle.loads(b"".join(data))
        #print("Done unpick")
        #data = conn.recv(40960000)
        #data = pickle.loads(data)
        if dataset == "mnist" and mode != "dfme":
            data = data.reshape((-1, 1, 28, 28))
        elif mode == "dfme":
            pass
        else:
            data = data.reshape((-1, 3, 32, 32))
        data = data.to(torch.float32)
        if cuda:
            data = data.cuda()
        preds = victim(data)
        num_queries += len(data)
        tdataset = [(a, 0) for a in data] # temp dataset
        adaptive_loader = DataLoader(
            tdataset,
            batch_size=64,
            shuffle=False)

        votes_victim = victim_model.inference(adaptive_loader, args2)
        datalength = len(votes_victim)
        for i in range(datalength):
            curvote = votes_victim[i][np.newaxis, :]
            max_num_query, dp_eps, partition, answered, order_opt = analysis.analyze_multiclass_confident_gnmax(
                votes=curvote,
                threshold=0,
                sigma_threshold=0,
                sigma_gnmax=args2.sigma_gnmax,
                budget=args2.budget,
                file=None,
                delta=args2.delta,
                show_dp_budget=False,
                args=args2
            )
            # print(f'dp_eps for vote {i}: {dp_eps[0]}')
            privacy_cost += dp_eps[0]
        #print('pate cost', privacy_cost)
        if mode != "dfme":
            preds = preds.cpu()

        # server

        bits = pow.get_leading_zero_bits_for_challenge_through_time(
            privacy_cost=privacy_cost, queries=num_queries)
        # print("bits", bits)

        xtype = 'bin'  # 'bin' or 'hex'
        resource = 'model-extraction-warning'
        challenge = generate_challenge(resource=resource, bits=bits)
        challengestr = pickle.dumps(challenge)
        conn.sendall(challengestr)
        stamp = conn.recv(4096)
        stamp = pickle.loads(stamp)

        is_correct = check(stamp=stamp, resource=resource, bits=bits,
                           check_expiration=DAY1, xtype=xtype)
        #is_correct = True

        #print("is correct", is_correct)

        if is_correct:
            predsstr = pickle.dumps(preds)
            conn.sendall(predsstr)
            # Only for larger batch sizes:
            # if mode == "dfme":
            #     time.sleep(0.01)
            #     str = "donesend"
            #     conn.sendall(str.encode())
    #i+=1
    #print(i)
conn.close()
s.close()

# This code is seperate and can be connected to from different attackers. Victim model returns logits with POW protocol.




