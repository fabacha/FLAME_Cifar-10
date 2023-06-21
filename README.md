# FLAME: Taming Backdoors in Federated Learning
This project implements FLAME found here https://arxiv.org/abs/2101.02281 using the CIFAR-10 dataset and a ResNet

# Credits

This project is based on code by Gehlar et al. the authors of SAFEFL https://eprint.iacr.org/2023/555

Their original codes and repository is here https://github.com/encryptogroup/SAFEFL


## Aggregation rules
The following aggregation rules have been implemented:

- [FedAvg](https://arxiv.org/abs/1602.05629)
- - 
- [FLAME](https://arxiv.org/abs/2101.02281)



## Attacks
To evaluate the robustness of the aggregation rules we also added the following attacks.

- [Label Flipping](https://proceedings.mlr.press/v20/biggio11.html)
- [Krum Attack](https://arxiv.org/abs/1911.11815)
- [Trim Attack](https://arxiv.org/abs/1911.11815)
- [Scaling Attack](https://arxiv.org/abs/2012.13995)
- [FLTrust Attack](https://arxiv.org/abs/2012.13995)
- [Min-Max Attack](https://par.nsf.gov/servlets/purl/10286354)
- [Min-Sum Attack](https://par.nsf.gov/servlets/purl/10286354)

The implementation of the attacks are all located in _attacks.py_ as individual functions.

To add a new attack the implementation can simply be added as a new function in this file. For attacks that are called during the aggregation the signature of the function 
must be the same format as the other attacks. This is because the attack function call in the training process is overloaded 
and which attack is called is only determined during runtime. 
The attack name must also be added to the get_byz function in _main.py_.
Attacks that only manipulate training data just need to be called before the training starts and don't need a specific signature.


## Multi-Party Computation
To run the MPC Implementation the [code](https://github.com/data61/MP-SPDZ) for [MP-SPDZ](https://eprint.iacr.org/2020/521) needs to be downloaded separately using the installation script _mpc_install.sh_.
The following protocols are supported:
- Semi2k uses 2 or more parties in a semi-honest, dishonest majority setting
- [SPDZ2k](https://eprint.iacr.org/2018/482) uses 2 or more parties in a malicious, dishonest majority setting
- [Replicated2k](https://eprint.iacr.org/2016/768.pdf) uses 3 parties in a semi-honest, honest majority setting
- [PsReplicated2k](https://eprint.iacr.org/2019/164.pdf) uses 3 parties in a malicious, honest majority setting


The project takes multiple command line arguments to determine the training parameters, attack, aggregation, etc. is used.
If no arguments are provided the project will run with the default arguments.
A description of all arguments can be displayed by executing:

```shell

python main_file.py -h

python main_file.py --niter 2000 --batch_size 64 --lr 0.25 --seed 123 --nruns 1 --test_every 10 --nbyz 2 --protocol psReplicated2k --players 4 --threads 8 --flame_epsilon 3000 --flame_delta 0.01 --aggregation flame --gpu 0 --net cnn --data CIFAR-10 --nworkers 100

```
# Requirements
The project requires the following packages to be installed:

- Python 3.8.13 
- Pytorch 1.11.0
- Torchvision 0.12.0
- Numpy 1.21.5
- MatPlotLib 3.5.1
- HDBSCAN 0.8.28
- Perl 5.26.2

All requirements can be found in the  _requirements.txt_.

# Environment.yml
You can create the environment using 

conda env create -f environment.yml

The MPC Framework MP-SPDZ was created by [Marcel Keller](https://github.com/data61/MP-SPDZ).

# License
[MIT](https://choosealicense.com/licenses/mit/)
