# Using PointNet++ to classify species in point clouds of single trees

This repository holds the code and instructions for a submission to the [Sensor-agnostic tree species classification using proximal laser scanning (TLS, MLS, ULS) and CNNs 🌳🌲💥🤖](https://github.com/stefp/Tr3D_species) hosted by the COST Action CA20118 ("3DForEcoTech").

This submission uses [PointNet++](https://github.com/charlesq34/pointnet2) with some adaptations for topographic point clouds, namely

- rotational augementation limited to the Z-Axis (vertical axis)

- sampling method to create point clouds of equal size (~16k points per sample)

- weighted loss calculation to account for class imbalance

The starting point for the code was the [PointNet++ Classification implementation by the pytorch-geometric team](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/pointnet2_classification.py).

For the leaderboard in the challenge, please visit [stefp's repository hosting the challenge](https://github.com/stefp/Tr3D_species#leader-board).

## Training performance

to tune the hyperparameters of the network, a validation set was created, amounting to 10% of the training set. The 10% were chosen randomly from each class, to ensure that every class also has samples in the validation set. No separate test set was created, as this is governed by the challenge.

## Tunable and tuned hyperparameters

The network comes with a number of tunable hyperparameters. An incomplete list follows below to give an idea of options to change the method.

- Number of Set Abstraction layers in PointNet++

- Number of neurons and layers in the MLPs of PointNet++

- Optimizer and its parameters, learning rate

- Loss function and esp. different weights

- Training data augmentation

- Batch size

- Point sampling strategies, handling of small point clouds (<16k points)

We trained the NN with different settings for all of these parameters, and observed the performance in macro F1 scores. The model with the best performance on the validation data during the training loop (not necessarily the last model) was selected.

## Running inference

To run the model on new data, ensure to download the weights of the model provided [here](). Then, change the parameters in `tree_classification_inference.py` to fit your input data (paths, path to the model) and run it.

This code assumes you have a GPU with CUDA enabled. It has been tested on Windows.


















