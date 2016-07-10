# General matchup modeling framework

## Overview

This general matchup modeling framework (or Matchup) is a software developed by [Shuo Chen](http://www.cs.cornell.edu/~shuochen/) from the Department of Computer Science, Cornell University during his PhD study. It learns from matchup or comparison data, and it can handle any associated player/item features and game/context features. Please see [our paper](coming soon) for more details. This program is granted free of charge for research and education purposes. However you must obtain a license from the author to use it for commercial purposes. Since it is free, there is no warranty for it.

## Build

A simple "make" will do. It will create a binary ../bin/Matchup, which serves for training and testing.

## Usage

### Format of the game record data

One can find sample dataset files (those used in our paper) under the datasets/ directory. The first six lines contain the meta information of the dataset, they are NumGames, NumTrainingGames, NumValidationGames, NumTestingGames, NumGameFeatures and NumPlayerFeatures. They take the format of "name: int\n". 

Starting from the seventh line to the rest of the file are the actually game records. Each record could be prefixed with a tag from "FOR\_TRAINING", "FOR\_VALIDATION" and "FOR\_TESTING", indicating what this line of record is for. In other cases, the prefix could be "FOR\_NONE" or simply missing. This suggests that this record is randomly assigned to training, validation and testing with probability of 0.5, 0.2 and 0.3.

The rest of the record is three sparse vectors delimited by '|'. They are game feature vector, winner's player feature vector and loser's player feature vector in that order. Each of the sparse vector is a set of none-zero components' indices and values. The index starts from 0.  

### Running the program

Matchup is used in the following format:

Matchup [options] data\_file model\_file

The "data\_file" is your training dataset. The "model\_file" is just a dummy here. I didn't implement I/O for the model, as it was not necessary for my experiments. You can implement it by yourself. All you need to do is to implement I/Os for the two data structures MachupModel and FullyConnectedLayer\_mult.   

An example is (assuming you are under /bin):

./Matchup -d 5 -l 0.001 -M 2 ../datasets/tennis/atp.txt mymodel.txt

Available options are:

```
-d						int               Dimensionality of the blade-chest layer (default 10)

-e						float             Error allowed for termination (default 1e-4)

-i						float             Learning rate (default 1e-3)

-l						float             Regularization coefficient (default 0.0)

-S						int               The seed for random number generator for creating different training, validation, and testing split and for randomly initializing the neural nets (default 0)

-m						int               Max number of iterations (default 1000)

-A						float             Adaptively increase the learning rate by this number if the improvement of the training objective function is too small (default 1.1, not recomeended to change if you run the code on our datasets)

-B						float             Adaptively decrease the learning rate by this number if the training objective function deteriorates (default 2.0, not recomeended to change if you run the code on our datasets)

-M						[0, 1, 2]         Model type. 0: use only player feature. 1: CONCAT model. 2: SPLIT model (default 0)  

-F						[NOACT, SIGMOID, TANH]     Acitvation function for neural net (default NOACT)

-L						[0, 1]            When turned on, only baseline models are trained (default 0)

-Y						[0, 1]            When turned on, only players' identities are used. For this flag to work properly, you need to make sure that the first index of each sparse player feature vectors in the training dataset stands for the player's identity (default 0) 

-s						float             The scaling factor of the matrices of the neural nets. Sometimes you need to have different values for different model type (default 0.01) 

-k						[0, 1]            When turned on, the prefixes in the training dataset will be ignored. The training, validation and testing partitions will be randomly generated (default 0)
```

### Outputs

The output of the software is the log on stdout. The log is human-readable, it contains information for each training iteration, validation/test log-likelihood and accuracy, results for the baselines.

## Datasets

These datasets are collected and processed by [Shuo Chen](http://www.cs.cornell.edu/~shuochen/) from multiple public sources on the internet. Every real-world dataset used in our paper is under /datasets. We do not own these data. Please cite each of the individual source if you use them for research or education purposes, and contact the source for any commercial use. Please see [our paper](http://csinpi.github.io/pubs/kdd16_chen.pdf) for details on the sources.   

## Bug Report

Please contact the author if you spot any bug in the software.

## References

If you use the software, please cite the following papers:

[Shuo Chen, Thorsten Joachims. Modeling Intransitivity in Matchup and Comparison Data. The 9th ACM International Conference on Web Search and Data Mining (WSDM)](http://www.cs.cornell.edu/~shuochen/pubs/wsdm16_chen.pdf)

[Shuo Chen, Thorsten Joachims. Predicting Matchups and Preferences in Context. The 22nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)](http://csinpi.github.io/pubs/kdd16_chen.pdf)
