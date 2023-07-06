#include "mpf.h"
// mpf -l [filename] [logsparsity] [p norm (optional)] // load in data, fit
// mpf -c [filename] [p norm (optional)] // load in data, fit, using cross-validation to pick best sparsity
// if [p norm] is specified, it will determine the exponent in the sparsity norm; the default is p=2 (Gaussian prior on coefficients); p=1 will select the Exponential (LASSO) prior on coefficients
// mpf -g [filename] [n_nodes] [n_obs] [beta] // generate data, save both parameters and data to files
// mpf -t [filename] [paramfile] [NN] // load in test data, fit, get KL divergence from truth
// mpf -o [filename_prefix] [NN] // load in data (_data.dat suffix), find best lambda using _params.dat to determine KL
// mpf -k [filename] [paramfile_truth] [paramfile_inferred] // load data, compare truth to inferred
// mpf -z [paramfile] [n_nodes]  // print out probabilities of all configurations under paramfile
// mpf -p [filename] [n_nodes] [paramfile] // print out log l of data given parameters
// mpf -s [paramfile] [n_nodes] [n_samples] // sample from the distribution many times

int main (int argc, char *argv[]) {
	double t0, running_logl, beta, *big_list, *truth, *inferred, logl_ans, glob_nloops, best_log_sparsity, kl_cv, kl_cv_sp, kl_true, kl_true_sp, ent, *best_fit;
	all *data;
	int i, ip, n, nn, thread_id, last_pos, in, j, count, pos, n_obs, n_nodes, kfold, num_no_na, tot_uniq, has_nans;
	sample *sav, **sav_list;
	cross_val *cv;
	unsigned long int config;
	char filename_sav[1000];
    FILE *fp, *fn;
	prob *p;
	
	t0=clock();
    
	if ((argc == 1) || (argv[1][0] != '-')) {

		printf("Greetings, Professor Falken. Please specify a command-line option.\n");
		
	} else {
		
        if (argv[1][1] == 'p') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
            
			n=atoi(argv[3]);
			truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
		    fp = fopen(argv[4], "r");
			for(j=0;j<n*(n+1)/2;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);
            
			init_params(data);
            data->big_list=truth;

            // running_logl=0;
            // for(i=0;i<data->uniq;i++) {
            //     config=0;
            //     for(j=0;j<data->n;j++) {
            //         if (data->obs[i]->config_base[j] > 0) {
            //             config += (1 << j);
            //         }
            //     }
            //     running_logl += data->obs[i]->mult*log_l(data, config, data->big_list, data->obs[i]->n_blanks, data->obs[i]->blanks);
            // }
            // printf("Total LogL for data, given parameters: %lf\n", running_logl);

			running_logl=0;
            i=0;
            for(i=0;i<data->uniq;i++) {
				config=0;
				for(j=0;j<data->n;j++) {
					if (data->obs[i]->config_base[j] > 0) {
						config += (1 << j);
					}
				}
				running_logl += data->obs[i]->mult*log_l_approx(data, config, data->big_list, data->obs[i]->n_blanks, data->obs[i]->blanks);
            }
			printf("Total LogL for data, given parameters, approx: %lf\n (%i)", running_logl, data->uniq);            
        }
		if (argv[1][1] == 'l') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
			            
            data->n_params=data->n*(data->n+1)/2;
    		data->best_fit=(double *)malloc(data->n_params*sizeof(double));
            count=0;
            for(i=0;i<data->n-1;i++) {
                for(j=(i+1);j<data->n;j++) {
                    if (i == 0) {
                        data->best_fit[count]=0.0;
                    } else {
                        data->best_fit[count]=0.0;
                    }
                    count++;
                }
            }
            for(i=count;i<data->n_params;i++) {
                data->best_fit[i]=0.0;
            }
            
            init_params(data);
            
			data->log_sparsity=atof(argv[3]);
			if (argc == 5) {
				data->p_norm=atof(argv[4]);
				printf("P norm set; p=%lf\n", data->p_norm);
			} else {
				data->p_norm=2.0;
				printf("P norm default; p=%lf\n", data->p_norm);
			}
			create_near(data, 1); 

            // printf("\n\nparams=[");
            // for(i=0;i<data->n_params;i++) {
            //     if (i < (data->n_params-1)) {
            //         printf("%.10e, ", data->big_list[i]);
            //     } else {
            //         printf("%.10e]\n", data->big_list[i]);
            //     }
            // }
			
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

			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);
			
            // running_logl=0;
            // for(i=0;i<data->uniq;i++) {
            //     config=0;
            //     for(j=0;j<data->n;j++) {
            //         if (data->obs[i]->config_base[j] > 0) {
            //             config += (1 << j);
            //         }
            //     }
            //                 if (data->n <= 20) {
            //                     running_logl += data->obs[i]->mult*log_l(data, config, data->big_list, data->obs[i]->n_blanks, data->obs[i]->blanks);
            //                 } else {
            //                     running_logl += data->obs[i]->mult*log_l_approx(data, config, data->big_list, data->obs[i]->n_blanks, data->obs[i]->blanks);
            //                 }
            //
            // }
            // printf("Total LogL for data, given parameters: %lf\n", running_logl);
		}

		if (argv[1][1] == 'c') { // cross validation
			// new idea : first find minimum without the NANs, then save that location, and "polish"
			
			data=new_data();
			read_data(argv[2], data);
			best_fit=NULL;
			nn=data->n; // number of nodes -- save this
						
			cv=(cross_val *)malloc(sizeof(cross_val));
			cv->filename=argv[2];
			cv->nn=nn; // atoi(argv[3]);
			cv->best_fit=best_fit;
			            
			if (argc == 4) {
				cv->p_norm=atof(argv[3]);
				printf("P norm set; p=%lf\n", cv->p_norm);
			} else {
				cv->p_norm=2.0;
				printf("P norm default; p=%lf\n", cv->p_norm);
			}
			best_log_sparsity=minimize_kl(cv, 0); // don't use fast version, just for safety
						
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
						
			data=new_data();
			read_data(argv[2], data);
			data->best_fit=best_fit; // will either be NULL (for the no NAN case, or the saved values)
			data->p_norm=cv->p_norm;
			
            
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, nn); // atoi(argv[3])
			
			printf("Now doing %i\n", data->m);			
			simple_minimizer(data);
			printf("%lf\n", data->k);
			printf("params=[");
			for(i=0;i<data->n_params;i++) {
				if (i < (data->n_params-1)) {
					printf("%.10e, ", data->big_list[i]);
				} else {
					printf("%.10e]\n", data->big_list[i]);
				}
			}
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "w+");
			for(j=0;j<data->n_params;j++) {
				fprintf(fp, "%.10e ", data->big_list[j]);
			}
		    fclose(fp);	
			
            // running_logl=0;
            // for(i=0;i<data->uniq;i++) {
            //     config=0;
            //     for(j=0;j<data->n;j++) {
            //         if (data->obs[i]->config_base[j] > 0) {
            //             config += (1 << j);
            //         }
            //     }
            //     running_logl += data->obs[i]->mult*log_l(data, config, data->big_list, data->obs[i]->n_blanks, data->obs[i]->blanks);
            // }
            // printf("Total LogL for data, given parameters: %lf\n", running_logl);
		}

		if (argv[1][1] == 'o') { // optimal lambda -- to be written
			
			cv=(cross_val *)malloc(sizeof(cross_val));
			
			// set up the data correctly...
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;

			cv->nn=atoi(argv[3]);
			
			// read in the true parameters correctly
			data=new_data();
			read_data(filename_sav, data);
			process_obs_raw(data);						
			init_params(data);
			cv->big_list_true=(double *)malloc(data->n_params*sizeof(double));
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_params.dat");
		    fp = fopen(filename_sav, "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(cv->big_list_true[j]));
			}
		    fclose(fp);
			
			p=(prob *)malloc(sizeof(prob));
			p->n=0;
			tot_uniq=0;
			for(i=0;i<data->uniq;i++) {
				if (data->obs[i]->n_blanks == 0) {
					p->n++;
					tot_uniq += data->obs[i]->mult;
				}
			}
			p->norm=-1;
			p->p=(double *)malloc(p->n*sizeof(double));
			ent=0;
			count=0;
			for(i=0;i<data->uniq;i++) {
				if (data->obs[i]->n_blanks == 0) {
					p->p[count]=data->obs[i]->mult;
					ent -= (data->obs[i]->mult*1.0/tot_uniq)*log((data->obs[i]->mult*1.0/tot_uniq))/log(2);
					count++;
				}
			}
			printf("NSB entropy of data: %lf\n", entropy_nsb(p));
			printf("Naieve entropy of data: %lf\n", ent);
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;
			
			best_log_sparsity=minimize_kl_true(cv);

			cv->kl_true=kl_holder(best_log_sparsity, (void *)cv);
			
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
			kl_true=cv->kl_true;
			kl_true_sp=best_log_sparsity;
			printf("KL at best log_sparsity: %lf\n", cv->kl_true);
			
			// now do CV
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_data.dat");
			cv->filename=filename_sav;
			best_log_sparsity=minimize_kl(cv, 0); // use fast version
			
			printf("Best log_sparsity CV: %lf\n", best_log_sparsity);
			
			data=new_data();
			read_data(cv->filename, data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, cv->nn);
												
			simple_minimizer(data);
			
			cv->kl_true=kl_holder(best_log_sparsity, (void *)cv);
			kl_cv=cv->kl_true;
			kl_cv_sp=best_log_sparsity;
			printf("KL at CV'd log_sparsity: %lf\n", cv->kl_true);
			
			printf("val=[%.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f, %.10f]\n", ent, entropy_nsb(p), kl_true, kl_true_sp, kl_cv, kl_cv_sp, kl_holder(-100, (void *)cv), -1000.0);
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
			
			cv=(cross_val *)malloc(sizeof(cross_val));
			cv->filename=argv[2];
			cv->nn=atoi(argv[4]);
			best_log_sparsity=minimize_kl(cv, 0); // use fast version
			
			printf("Best log_sparsity: %lf\n", best_log_sparsity);
			
			data=new_data();
			read_data(filename_sav, data);
			process_obs_raw(data);
						
			init_params(data);
			data->log_sparsity=best_log_sparsity;
			create_near(data, cv->nn);
												
			simple_minimizer(data);
			
			truth=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[3], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);
			
			printf("KL divergence from truth: %.10f\n", full_kl(data, data->big_list, truth));
			// now compare to the true distribution
		}
		if (argv[1][1] == 'k') {
			data=new_data();
			read_data(argv[2], data);
			process_obs_raw(data);
			init_params(data);
			
			truth=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[3], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);

			inferred=(double *)malloc(data->n_params*sizeof(double));
		    fp = fopen(argv[4], "r");
			for(j=0;j<data->n_params;j++) {
				fscanf(fp, "%le ", &(inferred[j]));
			}
		    fclose(fp);
            
			printf("KL: %lf\n", full_kl(data, inferred, truth));
			// n=atoi(argv[3]);
			// truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
			// 		    fp = fopen(argv[2], "r");
			// for(j=0;j<(n*(n+1)/2);j++) {
			// 	fscanf(fp, "%le ", &(truth[j]));
			// }
			// 		    fclose(fp);
			//
			// strcpy(filename_sav, argv[2]);
			// strcat(filename_sav, "_probs.dat");
			// compute_probs(n, truth, filename_sav);
			// now compare to the true distribution
		}
		if (argv[1][1] == 'z') {
			n=atoi(argv[3]);
			truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
		    fp = fopen(argv[2], "r");
			for(j=0;j<n*(n+1)/2;j++) {
				fscanf(fp, "%le ", &(truth[j]));
			}
		    fclose(fp);
			
			strcpy(filename_sav, argv[2]);
			strcat(filename_sav, "_probs.dat");
			
			compute_probs(n, truth, filename_sav);
		}
        if (argv[1][1] == 's') {
            n=atoi(argv[3]);
            truth=(double *)malloc((n*(n+1)/2)*sizeof(double));
            fp = fopen(argv[2], "r");
            for(j=0;j<n*(n+1)/2;j++) {
                fscanf(fp, "%le ", &(truth[j]));
            }
            fclose(fp);
            
			data=new_data();
            data->n=n;
            init_params(data);
            data->big_list=truth;

            strcpy(filename_sav, argv[2]);
            strcat(filename_sav, "_samples.dat");
            fn = fopen(filename_sav, "w+");

            for(i=0;i<atoi(argv[4]);i++) {
                mcmc_sampler(&config, 6, data);
        		for(ip=0;ip<n;ip++) {
        			if (config & (1 << ip)) {
        				fprintf(fn, "1");
        			} else {
        				fprintf(fn, "0");
        			}
        		}
                fprintf(fn, "\n");
            }
            fclose(fn);
            
        }
	}
	printf("Clock time: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	exit(1);
}