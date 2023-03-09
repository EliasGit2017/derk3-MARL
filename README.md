[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Derk3
=====

Derk3 is an agent for [Dr. Derk's Mutant Battlegrounds](https://derkgame.com/)
reinforcement learning evironment [Derk's Gym](https://gym.derkgame.com/). It
can serve as baseline for someone wanting to get started with the [AIcrowd
challenge](https://www.aicrowd.com/challenges/dr-derks-mutant-battlegrounds).

The most recent submission is
[here](https://www.aicrowd.com/challenges/dr-derks-mutant-battlegrounds/submissions/123185).

Environment
===========

Install dependencies and set up the development environment with
[Conda](https://docs.conda.io/en/latest/).

```sh
conda env create -f development.yml
conda activate derk3
```

First make sure the Derk's Gym environment runs by following the getting started
[instructions](https://gym.derkgame.com/).

Training
========

The training script will train an agent with the given hyperparameters using
self-play and output a checkpoint file. It is expected that you have a GPU
available for training.

```sh
./train.py --episodes 1000 --checkpoint checkpoint.pt parameters.json
```

Submission
==========

This repository is configured for submitting to the [AIcrowd
challenge](https://www.aicrowd.com/challenges/dr-derks-mutant-battlegrounds).

You should have [Large File Storage](https://git-lfs.github.com/) enabled in
your repository for storing the checkpoint. See the AIcrowd
[instructions](https://discourse.aicrowd.com/t/how-to-upload-large-files-size-to-your-submission/2304).

First edit [aicrowd.json](aicrowd.json) appropriately.. Then copy your
checkpoint into the agent directory, commit, and submit:

```sh
cp output/checkpoint.pt agent/
git add agent/checkpoint.pt
git commit -m "Update checkpoint"
git tag submission-v0.1
git push
git push origin submission-v0.1
```