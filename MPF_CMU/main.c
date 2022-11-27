#include "mpf.h"
// mpf -l [filename] [logsparsity] [NN] // load in data, fit
// mpf -c [filename] [logsparsity] [NN] // cross-validation 
// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files
// mpf -t [filename] [logsparsity] [NN] // load in test data, fit, get KL divergence from truth

int main (int argc, char *argv[]) {
	double t0, beta, *big_list, *truth;
	all *data;
	int i, last_pos, in, j, count, pos, n_obs, n_nodes, kfold, num_no_na;
	sample *sav;
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
			data=new_data();
			read_data(argv[2], data);
			
			// here's what we'll do -- we'll cycle through a bunch of samples where we leave out one data point
			num_no_na=0;
			for(i=0;i<data->m;i++) {
				if (data->obs_raw[i]->n_blanks == 0) {
					num_no_na++;
				}
			}
			
			printf("%i observations can be cross-validated.\n", num_no_na);
			if (num_no_na > 100) {
				num_no_na=100;
			}
			for(in=0;in<num_no_na;in++) {
				
				data=new_data();
				read_data(argv[2], data);
				data->m = data->m-1; // remove one data point

				pos=0;
				last_pos=0;
				count=0;
				while(count < (in+1)) {
					if(data->obs_raw[pos]->n_blanks == 0) { // if you see a good one, count it
						count++;
						last_pos=pos;
					}
					pos++; // move forward one unit
				}
				pos=last_pos;
				// printf("Location of %ith no-NA is %i\n", in, pos);

				sav=data->obs_raw[pos]; // the pointer to the data we'll leave out
				data->obs_raw[pos]=data->obs_raw[data->m]; //
				data->obs_raw[data->m]=sav;

				process_obs_raw(data);				
				init_params(data);
				data->log_sparsity=atof(argv[3]);
				create_near(data, atoi(argv[4]));

				// printf("%i data vectors; %i total; %i NNs\n", data->uniq, data->n_all, data->near_uniq);
									
				simple_minimizer(data);
			
				// printf("\n\nparams=[");
				// for(i=0;i<data->n_params;i++) {
				// 	if (i < (data->n_params-1)) {
				// 		printf("%.10e, ", data->big_list[i]);
				// 	} else {
				// 		printf("%.10e]\n", data->big_list[i]);
				// 	}
				// }
				
				// now we need to take that last data point, and compute the KL
				config=0;
				for(i=0;i<data->n;i++) {
					if (data->obs_raw[data->m]->config_base[i] > 0) {
						config += (1 << i);
					}
				}
				printf("LogL of left-out point: %lf\n", log_l(data, config, data->big_list));
			}
			
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