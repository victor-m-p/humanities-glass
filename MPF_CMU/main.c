#include "mpf.h"
// mpf -l [filename] [logsparsity] [NN] // load in data, fit
// mpf -c [filename] [NN] // load in data, fit, using cross-validation to pick best sparsity
// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files
// mpf -t [filename] [logsparsity] [NN] // load in test data, fit, get KL divergence from truth
// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL

int main (int argc, char *argv[]) {
	double t0, beta, *big_list, *truth, logl_ans, glob_nloops, best_log_sparsity;
	all *data;
	int i, thread_id, last_pos, in, j, count, pos, n_obs, n_nodes, kfold, num_no_na;
	sample *sav;
	cross_val *cv;
	unsigned long int config;
	char filename_sav[1000];
    FILE *fp;
	
	t0=clock();

	if ((argc == 1) || (argv[1][0] != '-')) {

		printf("Greetings, Professor Falken. Please specify a command-line option.\n");
		
	} else {
		
		if (argv[1][1] == 'l') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=atof(argv[3]);
			create_near(data, atoi(argv[4]));
			
			printf("%i data vectors; %i total; %i NNs\n", data->uniq, data->n_all, data->near_uniq);
									
			simple_minimizer(data);
			
			printf("\n\nparams=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}
		}

		if (argv[1][1] == 'c') { // cross validation
			
			cv=(cross_val *)malloc(sizeof(cross_val));
			cv->filename=argv[2];
			cv->nn=atoi(argv[3]);
			best_log_sparsity=minimize_kl(cv);
			
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
			
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, cv->nn);
												
			simple_minimizer(data);
			
			printf("\n\nparams=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}
			
		}

		if (argv[1][1] == 'o') { // optimal lambda -- to be written
			
			// cv=(cross_val *)malloc(sizeof(cross_val));
			// cv->filename=argv[2];
			// cv->nn=atoi(argv[3]);
			// best_log_sparsity=minimize_true_kl(cv);
			//
			// printf("Best log_sparsity: %lf\n", best_log_sparsity);
			//
			// data=new_data();
			// read_data(argv[2], data);
			// process_obs_raw(data);
			//
			// init_params(data);
			// data->log_sparsity=best_log_sparsity;
			// create_near(data, cv->nn);
			//
			// simple_minimizer(data);
			//
			// printf("\n\nparams=[");
			// for(i=0;i<data->n_params;i++) {
			// 	if (i < (data->n_params-1)) {
			// 		printf("%.10e, ", data->big_list[i]);
			// 	} else {
			// 		printf("%.10e]\n", data->big_list[i]);
			// 	}
			// }
			
		}
		
		if (argv[1][1] == 'g') {
			n_nodes=atoi(argv[3]);
			n_obs=atoi(argv[4]);
			beta=atof(argv[5]);
	
			data=new_data();
			data->n=n_nodes;
			data->m=n_obs;
			init_params(data);
			
			for(i=0;i<data->n_params;i++) {
				data->big_list[i]=gsl_ran_gaussian(data->r, 1.0)*beta;
			}
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
		    fp = fopen(filename_sav, "w+");
		    fprintf(fp, "%i\n%i\n", data->m, data->n);
			for(j=0;j<data->m;j++) {
				config=gsl_rng_uniform_int(data->r, (1 << data->n));
				mcmc_sampler(&config, 1000, data);
				for(i=0;i<data->n;i++) {
					if (config & (1 << i)) {
						fprintf(fp, "1");
					} else {
						fprintf(fp, "0");
					}
				}
				fprintf(fp, " 1.0\n");
			}
		    fclose(fp);

			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);
		}
		if (argv[1][1] == 't') {
			data=new_data();
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			read_data(filename_sav, data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=atof(argv[3]);
			create_near(data, atoi(argv[4]));
			
			printf("%i data vectors; %i total; %i NNs\n", data->uniq, data->n_all, data->near_uniq);
									
			simple_minimizer(data);
			
			printf("\n\nparams=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}

			truth=(double *)malloc(data->n_params*sizeof(double));
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);

			printf("\n\ntruth=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", truth[i]);
				} else {
					printf("%.10e]\n", truth[i]);
				}
			}
			
			printf("KL divergence: %.10f\n", full_kl(data, data->big_list, truth));
			// now compare to the true distribution
		}
	}
	printf("Clock time: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	exit(1);
}