# Increasing the Cost of Model Extraction with Calibrated Proof of Work

*Keywords:* model extraction, privacy, proof of work, differential privacy,
information theory, attacks, defenses

*TL;DR:* A novel model extraction defense which uses Proof of Work techniques calibrated with the privacy cost to increase the time an attacker has to spend querying.

*Abstract:* In model extraction attacks, adversaries can steal a machine learning model exposed via a public API by repeatedly querying it and adjusting their own model based on obtained predictions. To prevent model stealing, existing defenses focus on detecting malicious queries, truncating, or distorting outputs, thus necessarily introducing a tradeoff between robustness and model utility for legitimate users. Instead, we propose to impede model extraction by requiring users to complete a proof-of-work before they can read the model's predictions. This deters attackers by greatly increasing (even up to 100x) the computational effort needed to leverage query access for model extraction. Since we calibrate the effort required to complete the proof-of-work to each query, this only introduces a slight overhead for regular users (up to 2x). To achieve this, our calibration applies tools from differential privacy to measure the information revealed by a query. Our method requires no modification of the victim model and can be applied by machine learning practitioners to guard their publicly exposed models against being easily stolen

*Paper*: https://openreview.net/forum?id=EAy7C1cgE1L

## Description of the code

The `main.py` file contains the skeleton our code uses. `main_model_extraction.py` is the main file used to run model extraction attacks and measure the relevant cost metrics. It is called from the `main.py` code. Default parameters used in the code
are in the `parameters.py` file. For the CIFAR10 and SVHN victim models, we use the models from the Data Free Model Extraction repository (https://github.com/cake-lab/datafree-model-extraction). For MNIST, we use the MnistNet Architecture on which we can train a model locally using the script `experiments/train.sh`.  

### Examples of the pipeline:

The bash code for running some example scripts can be found in the `scripts` folder. Note that experiments on different datasets 
and with different modes are run in a similar way by changing the relevant parameters. We list the different types of attacks and how 
they are run. 

1. Standard, Entropy, EntropyRev, WorstCase, Knockoff, CopyCat, Jacobian and other Active Learning Methods

All of these attacks can be run using a similar script and are all based primarily in the 
`main_model_extraction.py` file. Two example scripts for standard and knockoff on the MNIST dataset are in the `scripts` folder. 

For Jacobian, the code for the attack itself is located in `jacobian.py` and relevant hyperparameters and ways of running the experiment can be adjusted there.
The output logs containing the costs and the accuracies reached are saved in the folder `adaptive-model`.

2. MixMatch:

The code for MixMatch can be found in the `MixMatch-pytorch` folder. This code has been adapted for our needs from a publicly available repository at https://github.com/YU1ut/MixMatch-pytorch. A sample script to run this code for the MNIST dataset can be found in the folder.     



3. Data Free Model Extraction

The code to run this along with example scripts are located in the folder `dfme`. This code has been adapted from the official github repository of the paper (https://arxiv.org/abs/2011.14779, https://github.com/cake-lab/datafree-model-extraction). Sample scripts to run this attack can for the MNIST or CIFAR10 datasets can be found at `dfme/dfme/mnisttest.sh` and `dfme/dfme/cifartest.sh` respectively.   


4. Knockoff Nets from ART 

To run the Knockoff Nets attack with the adaptive version of the attack, the file `adversarial-robustness-toolbox/knockoff_nets_run.py` can be run with the appropriate parameters. This code is based on the ART library https://github.com/Trusted-AI/adversarial-robustness-toolbox.  
   
5. Proof of Work

The code for proof of work can be found in the file `proof_of_work.py` in the `pow` folder. The file `proof_of_work_test.py` demonstrates how this is applied. Csv files containing the time and privacy costs for various attack methods and the MNIST, SVHN and CIFAR10 datasets can also be found in the same folder. 

The Csv files starting with `time_privacy_cost_` are for calibrating the proof of work with times and privacy costs for a standard user. The files starting with `model_extraction_attack_accuracy_` contain times and privacy costs for different attack methods and are used for evaluating the effect of the PoW on the different methods. Note that the `diff queries` column corresponds to the number of batches of size 64 required for the numbers in that row e.g. `diff_queries=16` corresponds to 1000 queries. The times can be calculated based on Table 5 in the paper. For example for 1000 queries (16 batches of 64 queries) with a CIFAR10 victim model, the time will be 16 * 1.03 = 16.48 seconds.        

6. Server Client Setup 

We also include a server client set up to simulate this defense in a real world scenario. The file `server.py` starts a server and loads the relevant victim model. Setting the parameter `--useserver` to True chooses this as the method used to answer the queries with the proof of work applied based on the privacy cost. 
Note that both `server.py` and the program for the attack must be running simultaneously for the communication to be possible. 


### Citing this work
If you use this repository for academic research, you are highly encouraged (though not required) to cite our paper:
```
@inproceedings{
dziedzic2022increasing,
title={Increasing the Cost of Model Extraction with Calibrated Proof of Work},
author={Adam Dziedzic and Muhammad Ahmad Kaleem and Yu Shen Lu and Nicolas Papernot},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=EAy7C1cgE1L}
}
```