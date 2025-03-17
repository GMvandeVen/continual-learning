#!/usr/bin/env bash




########### ICLR 2025 Blogpost ###########

python3 ICLRblogpost/compare_FI.py --seed=1 --n-seeds=30 --experiment=splitMNIST --scenario=task
python3 ICLRblogpost/compare_FI.py --seed=1 --n-seeds=30 --experiment=CIFAR10 --scenario=task --reducedResNet --iters=2000 --lr=0.001




########### NeurIPS 2022 Tutorial ###########

python NeurIPStutorial/compare_for_tutorial.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=task
python NeurIPStutorial/compare_for_tutorial.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=domain
python NeurIPStutorial/compare_for_tutorial.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=class




########### Three Types of Incremental Learning (2022, Nat Mach Intell) ###########

## MNIST

./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=task
./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=domain
./compare_hyperParams.py --seed=1 --experiment=splitMNIST --scenario=class

./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=task
./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=domain
./compare.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=class

./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=task --tau-per-budget
./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=domain --tau-per-budget
./compare_replay.py --seed=2 --n-seeds=5 --experiment=splitMNIST --scenario=class --tau-per-budget


## Pre-training on CIFAR-10

./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=1
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=2
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=3
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=4
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=5
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=6
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=7
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=8
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=9
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=10
./main_pretrain.py --experiment=CIFAR10 --epochs=100 --augment --convE-stag=e100 --seed-to-stag --seed=11


## CIFAR-100

./compare_hyperParams.py --seed=1 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --no-fromp
./compare_hyperParams.py --seed=1 --experiment=CIFAR100 --scenario=domain --pre-convE --freeze-convE --seed-to-ltag --no-fromp
./compare_hyperParams.py --seed=1 --experiment=CIFAR100 --scenario=class --pre-convE --freeze-convE --seed-to-ltag --no-fromp

./compare.py --seed=2 --n-seeds=10 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --no-fromp --seed-to-ltag
./compare.py --seed=2 --n-seeds=10 --experiment=CIFAR100 --scenario=domain --pre-convE --freeze-convE --no-fromp --seed-to-ltag
./compare.py --seed=2 --n-seeds=10 --experiment=CIFAR100 --scenario=class --pre-convE --freeze-convE --no-fromp --seed-to-ltag --eval-s=10000

./compare_replay.py --seed=2 --n-seeds=5 --experiment=CIFAR100 --scenario=task --pre-convE --freeze-convE --seed-to-ltag --no-fromp
./compare_replay.py --seed=2 --n-seeds=5 --experiment=CIFAR100 --scenario=domain --pre-convE --freeze-convE --seed-to-ltag --no-fromp
./compare_replay.py --seed=2 --n-seeds=5 --experiment=CIFAR100 --scenario=class --pre-convE --freeze-convE --seed-to-ltag --no-fromp


## Permuted MNIST

./compare_hyperParams.py --seed=1 --experiment=permMNIST --scenario=task --fisher-n=1000 --no-xdg --no-fromp --no-bir
./compare_hyperParams.py --seed=1 --experiment=permMNIST --scenario=task --singlehead --fisher-n=1000 --no-reg --no-fromp --no-bir
./compare_hyperParams.py --seed=1 --experiment=permMNIST --scenario=task --singlehead --xdg --gating-prop=0.6 --fisher-n=1000 --no-xdg --no-fromp --no-bir
./compare_hyperParams.py --seed=1 --experiment=permMNIST --scenario=domain --fisher-n=1000 --no-fromp --no-bir

./compare.py --seed=2 --n-seeds=5 --experiment=permMNIST --scenario=task --no-context-spec --no-fromp --no-bir --no-agem --fisher-n=1000
./compare.py --seed=2 --n-seeds=5 --experiment=permMNIST --scenario=task --singlehead --xdg --gating-prop=0.6 --no-context-spec --no-fromp --no-bir --no-agem --fisher-n=1000
./compare.py --seed=2 --n-seeds=5 --experiment=permMNIST --scenario=domain --no-fromp --no-bir --no-agem --fisher-n=1000


## Task-free Split MNIST

./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=task
./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=domain
./compare_hyperParams_task_free.py --seed=1 --experiment=splitMNIST --scenario=class

./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=task --gating-prop=0.45 --c=10.
./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=domain --c=10.
./compare_task_free.py --seed=2 --n-seeds=20 --experiment=splitMNIST --scenario=class --c=10.

#