## Overview

Code for "Inferring Cultural Landscapes with the Inverse Ising Model" (https://www.mdpi.com/1099-4300/25/2/264/html)
has been migrated to https://github.com/victor-m-p/cultural-landscapes. See that repository for documentation.
For related work on the DRH, see also: https://github.com/victor-m-p/cognitive-attractors. 
This workspace contains the most up-to-date implementation of ```/MPF_CMU``` and some core preprocessing of the DRH data
used across different analyses efforts. 

Victor Møller Poulsen and Simon DeDeo

## 2026 Claude/Codex Analysis

In November 2026 the MPF code and methods were re-examined by Claude (Anthropic), with an independent review by Codex (OpenAI). The full account is [CLAUDE_IMPROVEMENTS_SUMMARY.md](https://github.com/victor-m-p/cultural-landscapes/blob/main/CLAUDE_IMPROVEMENTS_SUMMARY.md) in the cultural-landscapes repository. For this repository:

* **A revised program is in [`MPF_CMU/src`](MPF_CMU/src)**: the same code as in cultural-landscapes, self-contained (C sources, Makefile, tests; no external libraries). `cd MPF_CMU/src && make && make test`. It corrects the treatment of missing data, implements two- and three-flip neighbourhoods, fixes the defects listed below, and is 50–150 times faster. Use it for new work.
* **The C files directly in `MPF_CMU` have not been changed.** They are the 2023 development version with which the experiments in this repository were run, so it is no longer true that this is the most up-to-date implementation. That version has things the revised program does not: a penalty with a p-norm (p=1 for an L1 penalty), cross-validation that also holds out rows with missing entries, and the `-p` and `-s` modes; its last argument to `-l` and `-c` is the p-norm, not the neighbourhood. It also has the defects found in the published code: a biased objective for missing data ("Partial-MPF"), wrong states for more than 31 nodes, a loop variable shared between threads during cross-validation, and a minimiser whose failure to converge goes unreported.
* **Hidden nodes need particular care.** The bias of Partial-MPF is at its worst for a node that is missing in every row, which is how the hidden-node fits here were set up. On exact data from models with five visible nodes and one hidden node, and no regularisation, the 2023 objective is not stationary at the true parameters, and an optimiser started there ends with the couplings to the hidden node 20 to 230 times their true size; the revised objectives are stationary at the truth. The hidden-node results here (`MPF_CMU/HIDDEN_FITS*`, `MPF_CMU/HIDDEN_TRIALS`) should be checked against the revised program before they are relied on. The revised program fits such data at a fixed regularisation strength (`-l`, with `--start` to break the symmetry of the hidden nodes), but its cross-validation needs rows without missing entries, so it cannot yet choose the regularisation strength for them.

### ```/MPF_CMU``` 

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

<!-- LICENSE -->
## License
Distributed under the MIT License. See `LICENSE.txt` for more information.

## Contact

Simon DeDeo (for ```/MPF_CMU``` questions):
* Twitter: [@LaboratoryMinds](https://twitter.com/LaboratoryMinds)
* Github: [@simon-dedeo](https://github.com/simon-dedeo)
* Mail: sdedeo@andrew.cmu.edu

Victor Poulsen (for other questions): 
* Twitter: [@vic_moeller](https://twitter.com/vic_moeller) 
* GitHub: [@victor-m-p](https://github.com/victor-m-p)
* Mail: victormoeller@gmail.com

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
We are particularly grateful to the creaters of ```conIII``` and the maintainers and creators of the ```Database of Religious History (DRH)```.

* [ConIII](https://github.com/eltrompetero/coniii)
* [Database of Religious History (DRH)](https://religiondatabase.org/landing/)

## FUNDING

This work used the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant number ACI-1548562. Specifically, it used the Bridges-2 system, which is supported by NSF award number ACI-1928147, at the Pittsburgh Supercomputing Center (PSC), under grant HUM220003. This work was supported in part by the Survival and Flourishing Fund.
