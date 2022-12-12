## Overview

Pre-review code for "Inferring Cultural Landscapes with the Inverse Ising Model"

Victor Møller Poulsen and Simon DeDeo

submitted to *Entropy*, 9 December 2022

This is *pre-review* code, not intended for general public circulation. We would be very grateful for comments, questions, and thoughts.

We are very happy to take requests. Our experience has been that iterating with potential users is the best way to make our code and data clear, reusable, and efficient to use. On re-submission and review, we will release general code, intended for consumer use, and a full guide.

## Components

### ```/data```

Processed DRH data can be found in data/mdl_final, with additional information in data/reference.

### 

### MPF code 

MPF_CMU contains the optimized C code to implement all of the extensions and modifications to MPF described in our paper. Many of the simulations were carried out on the Bridges2 Pittsburgh Supercomputing Center system, and for speed they are set up to use multiple cores with OpenMP. You will want to adjust the Makefile to compile on your local system. Note that the compiler that ships with the new Mac M1s does not support OpenMP; you will need to install a (non-Apple) clang compiler.

Some of this code is compute intensive; on a Mac M1 with 10 cores, for example, it takes a few seconds to fit n=20 with a few hundred observations, and a few minutes to do the same with Cross Validation.

MPF has seven different modes:

// mpf -l [filename] [logsparsity] [NN] // load in data, fit

Given a data file, a log-sparsity choice, and a specification of \mathcal{N}_i, produces a parameter fit file. e.g.,

./mpf -l test.dat 1.0 1

will fit using a lambda equal to 10^1.0, and the \mathcal{N}_1 strategy.

// mpf -c [filename] [NN] // load in data, fit, using cross-validation to pick best sparsity

The same as above, but does cross-validation to find the best value of lambda.

// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files

Generates a simulated dataset, with n_nodes nodes, and n_obs observations, with J and h drawn from a Gaussian with RMS equal to beta.

// mpf -t [filename] [paramfile] [NN] // load in test data, fit, get KL divergence from truth

If you know the true parameters (e.g., because you previously ran mpf -g), this will fit data, and tell you how well you did.

// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL

If you know the true parameters, this will tell you the secret best value of lambda.

// mpf -k [filename] [paramfile_truth] [paramfile_inferred] // load data, compare truth to inferred

If you have previously fit some data, but you also know the true parameters, this will tell you how well you did.

// mpf -z [paramfile] [n_nodes]  // print out probabilities of all configurations under paramfile

If you have a parameter file, this will print out all the configurations, the energy for each config, and the probability.

* FUNDING

This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562. Specifically, it used 509 the Bridges-2 system, which is supported by NSF award number ACI-1928147, at the Pittsburgh 510 Supercomputing Center (PSC), under grant HUM220003. This work was supported in part by the Survival and Flourishing Fund.