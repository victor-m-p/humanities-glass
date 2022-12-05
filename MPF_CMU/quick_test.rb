// mpf -l [filename] [logsparsity] [NN] // load in data, fit
// mpf -c [filename] [NN] // load in data, fit, using cross-validation to pick best sparsity
// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files
// mpf -t [filename] [paramfile] [NN] // load in test data, fit, get KL divergence from truth
// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL
// mpf -k [filename] [paramfile_truth] [paramfile_inferred] // load data, compare truth to inferred
// mpf -z [paramfile] [n_nodes]  // print out probabilities of all configurations under paramfile

./mpf -g TEST/test 20 512 0.5

./mpf -z TEST/test_params.dat 20

./mpf -c TEST/test_data.dat 1

./mpf -z TEST/test_data.dat_params.dat 20

cp TEST/test_data.dat_params.dat_probs.dat TEST/test_data.dat_params.dat_probs_CV.dat
cp TEST/test_data.dat_params.dat TEST/test_data.dat_params_CV.dat

./mpf -l TEST/test_data.dat -100 1

./mpf -z TEST/test_data.dat_params.dat 20

./mpf -k TEST/test_data.dat TEST/test_params.dat TEST/test_data.dat_params_CV.dat
./mpf -k TEST/test_data.dat TEST/test_params.dat TEST/test_data.dat_params.dat

