#include "mpf.h"
// mpf -l [filename] [logsparsity] [compute out of sample KL] // load in data, simulate
// mpf -s n_obs n_nodes beta iter nn // simulate a random system with n_obs, n_nodes, and beta (scaling parameter for J_ij and h_i)
// mpf -o n_obs n_nodes beta iter sparsity nn // simulate a random system with n_obs, n_nodes, and beta (scaling parameter for J_ij and h_i)

int main (int argc, char *argv[]) {
	double t0;
	samples *data;
	int i, count, sp, j, nn, pos, upsample, sp_tot=20;
	double beta, running, *saved_list, *final_list, **big_sav;
	double ent, *sp_sav, *kl, *kl_sp, kl_true, kl_guess, kl_best, kl_best_loc, *j_var, sp_d;
	prob *p;
	
	t0=clock();

	if ((argc == 1) || (argv[1][0] != '-')) {

		printf("Greetings, Professor Falken. Please specify a command-line option.\n");
		
	} else {
		
		data=new_data(); // setting up the system 

		if (argv[1][1] == 'l') {
			printf("Loading in from %s\n", argv[2]);
			load_data(argv[2], data);	
			init_params(data, 1); // initialize the J_ij and h_i guesses			
			sort_and_filter(data); // remove duplicates from the data, create the nearest neighbour list
			printf("%i data points, %i questions.\n", data->m, data->n);
			printf("log10 Sparsity is %lf.\n", atof(argv[3]));
			
			data->sparsity=0;
			for(i=0;i<data->uniq;i++) {
				for(j=0;j<data->near_uniq;j++) {
					data->sparsity += data->mult[i]*data->prox[i][j];
				}
			}
			data->sparsity *= (1.0/data->n_params); // divide by the total number of parameters?
			data->sparsity *= exp(atof(argv[3])*log(10));
			printf("Rescaled Sparsity is %lf.\n", data->sparsity);
			
			// now compute NSB entropy
			p=(prob *)malloc(sizeof(prob));
			p->n=data->uniq; // n becomes uniq 
			p->norm=-1; // norm -1 
			p->p=(double *)malloc(data->uniq*sizeof(double));
			ent=0;
			for(i=0;i<data->uniq;i++) { // for each uniq
				p->p[i]=data->mult[i]; // how many times does the state appear?
				// entropy below is uniq
				ent -= (data->mult[i]*1.0/data->m)*log((data->mult[i]*1.0/data->m))/log(2);
			}
			printf("NSB entropy of data: %lf\n", entropy_nsb(p)); // check this
			printf("Naieve entropy of data: %lf\n", ent); //
			
			create_neighbours(data, 3); // do up to the 3-bit nearest neighbours (thought this was nn?)
			
			simple_minimizer(data); // minimize...
			
			printf("RMS of parameters: %14.12lf\n", sqrt(gsl_stats_variance(data->big_list, 1, data->n_params)));
			
			printf("parameter_list=");
			pretty(data->big_list, data->n_params);
			
			if (argc == 5) {
				printf("Computing in-sample KL of final model (may take a while)\n");
				printf("Estimated log-L of data, given inferred parameters: %17.12lf\n", sample_states(data, data->n*10, 100000, data->big_list));				
			}
			
			saved_list=NULL;
		}
		if (argv[1][1] == 's') {
			beta=atof(argv[4]); 
						
			blank_system(data, atoi(argv[2]), atoi(argv[3])); // m and n
			
			init_params(data, 0);
			
			for(i=0;i<data->n_params;i++) {
				data->big_list[i] *= beta;
			}
			if (argc >= 7) { // hmmm? 
				nn=atoi(argv[6]); // nearest neighbor 
			} else {
				nn=3; // nearest neighbor default 
			}
			
			// reset and resimulate			
			// not sure whether we can avoid this if we are using a class 
			for(i=0;i<data->m;i++) {
				for(j=0;j<data->n;j++) {
					data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
				}
				mcmc_sampler(data, i, data->n*atoi(argv[5]), data->big_list); // n*iter
			}

			// overview of data						
			printf("data=[");
			for(i=0;i<data->m;i++) {
				printf("[");
				for(j=0;j<data->n-1;j++) {
					printf("%i, ", data->obs[i][j]);
				}
				printf("%i]", data->obs[i][data->n-1]);
				if (i < (data->m-1)) {
					printf(",");
				}
			}
			printf("]\n");

			// not really sure whether we use this copy...?
			saved_list=(double *)malloc(data->n_params*sizeof(double));
			for(i=0;i<data->n_params;i++) {
				saved_list[i]=data->big_list[i];
			}

			init_params(data, 1); // 1 for correlations // now pretend it's real data, and re-initialize the J_ij and h_i
			sort_and_filter(data); // remove duplicates from the data
			
			// now compute NSB entropy
			// check up on NSB
			p=(prob *)malloc(sizeof(prob));
			p->n=data->uniq;
			p->norm=-1;
			p->p=(double *)malloc(data->uniq*sizeof(double));
			ent=0;
			for(i=0;i<data->uniq;i++) {
				p->p[i]=data->mult[i];
				ent -= (data->mult[i]*1.0/data->m)*log((data->mult[i]*1.0/data->m))/log(2);
			}
			printf("NSB entropy: %lf\n", entropy_nsb(p));
			printf("Naieve entropy: %lf\n", ent);
			printf("Doing fits...\n");
			// if (saved_list != NULL) {
			// 	printf("Estimated log-L of data, given true: %17.12lf\n", sample_states(data, data->n*10, 100000, saved_list)); // many samples, a little correlated
			// }

			// here Simon runs for different nn values
			// still cannot see that the nn input does anything
			// seems like it always runs 1, 2, 3 (nn) -- CHECK THIS. 
			for(nn=1;nn<=3;nn++) {
				
				big_sav=(double **)malloc(sp_tot*sizeof(double *));
				sp_sav=(double *)malloc(sp_tot*sizeof(double));
			
				kl=(double *)malloc(sp_tot*sizeof(double));
				kl_sp=(double *)malloc(sp_tot*sizeof(double));
				j_var=(double *)malloc(sp_tot*sizeof(double));
			
				count=0;
				for(sp=0;sp<sp_tot;sp++) {
					create_neighbours(data, nn); // key function
				
					data->sparsity=0;
					for(i=0;i<data->uniq;i++) {
						for(j=0;j<data->near_uniq;j++) {
							data->sparsity += data->mult[i]*data->prox[i][j];
						}
					}
					data->sparsity *= (1.0/data->n_params); // divide by the total number of parameters?
					kl_sp[count]=(-3+(1.0*sp/sp_tot)*5);
					data->sparsity *= exp(kl_sp[count]*log(10));
					if (sp == 0) {
						data->sparsity=0;
					}
					
					sp_sav[count]=data->sparsity;
					simple_minimizer(data); // minimize...
					// printf("Sp %lf NN %i Estimated log-L of data, given inferred: %17.12lf\n", data->sparsity, nn, sample_states(data, data->n*10, 100000, data->big_list)); // many samples, a little correlated
					big_sav[count]=(double *)malloc(data->n_params*sizeof(double));
					for(i=0;i<data->n_params;i++) {
						big_sav[count][i]=data->big_list[i];
					}
					count++;
					// printf("%lf\n", data->sparsity);
				}
				// printf("Fits done\n");
				
				//need to clean the neighbour sets
				// for(i=0;i<data->near_uniq;i++) {
				// 	free(data->near[i]);
				// }
				// free(data->near);
				// free(data->near_ok);
				// for(i=0;i<data->uniq;i++) {
				// 	free(data->prox[i]);
				// }
				// free(data->prox);
				// data->prox=NULL;
				// for(i=0;i<data->uniq;i++) {
				// 	free(data->obs[i]);
				// }
				// free(data->obs);
				// data->near_uniq=0;

				// data->m *= 10;
				// data->obs=(int **)malloc(data->m*sizeof(int *));
				// for(i=0;i<data->m;i++) {
				// 	data->obs[i]=(int *)malloc(data->n*sizeof(int));
				// 	for(j=0;j<data->n;j++) {
				// 		data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
				// 	}
				// 	mcmc_sampler(data, i, data->n*atoi(argv[5]), saved_list);
				// }
				// sort_and_filter(data); // remove duplicates from the data

				// printf("Out of sample\n");
				// printf("RMS of J/h: %14.12lf\n", sqrt(gsl_stats_variance(saved_list, 1, data->n_params)));
			
				// kl_true=sample_states(data, data->n*10, 100000, saved_list);
				// printf("Estimated log-L of data, out of sample, given true: %17.12lf\n\n", kl_true); // many samples, a little correlated
				count=0;	
				for(sp=0;sp<sp_tot;sp++) {
					kl[count]=full_kl(data, big_sav[count], saved_list);
					j_var[count]=sqrt(gsl_stats_variance(big_sav[count], 1, data->n_params));
					// printf("Sp %lf (%lf) NN %i Estimated KL of data, given inferred: %17.12lf\n", kl_sp[count], sp_sav[count], nn, kl[count]); // many samples, a little correlated
					// printf("RMS of J/h: %14.12lf\n", sqrt(gsl_stats_variance(big_sav[count], 1, data->n_params)));
					count++;	
					// printf("\n");
				}
				kl_best=-1e300;
				kl_best_loc=-1e12;
				for(sp=0;sp<sp_tot;sp++) {
					if (kl[sp] > kl_best) {
						kl_best=kl[sp];
						kl_best_loc=kl_sp[sp];
					}
				}
				// printf("Best location is %lf, gives %lf\n", kl_best_loc, kl_best);
			
				printf("sp%i=[", nn);
				for(sp=0;sp<sp_tot;sp++) {
					printf("[%lf, %lf]", kl_sp[sp], kl[sp]);
					if (sp != (sp_tot-1)) {
						printf(",");
					}
				}
				printf("]\n");
			}
			
		}
		if (argv[1][1] == 'o') {
			beta=atof(argv[4]);						
			blank_system(data, atoi(argv[2]), atoi(argv[3])); // m and n
			init_params(data, 0);
			for(i=0;i<data->n_params;i++) {
				data->big_list[i] *= beta;
			}
			sp_d=atof(argv[6]);
			nn=atoi(argv[7]);
			
			// reset and resimulate			
			for(i=0;i<data->m;i++) {
				for(j=0;j<data->n;j++) {
					data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
				}
				mcmc_sampler(data, i, data->n*atoi(argv[5]), data->big_list);
			}
			saved_list=(double *)malloc(data->n_params*sizeof(double));
			for(i=0;i<data->n_params;i++) {
				saved_list[i]=data->big_list[i];
			}

			init_params(data, 1); // 1 for correlations // now pretend it's real data, and re-initialize the J_ij and h_i
			sort_and_filter(data); // remove duplicates from the data

			p=(prob *)malloc(sizeof(prob));
			p->n=data->uniq;
			p->norm=-1;
			p->p=(double *)malloc(data->uniq*sizeof(double));
			ent=0;
			for(i=0;i<data->uniq;i++) {
				p->p[i]=data->mult[i];
				ent -= (data->mult[i]*1.0/data->m)*log((data->mult[i]*1.0/data->m))/log(2);
			}
			printf("NSB entropy: %lf\n", entropy_nsb(p));
			printf("Naieve entropy: %lf\n", ent);

			create_neighbours(data, nn);
			
			data->sparsity=0;
			for(i=0;i<data->uniq;i++) {
				for(j=0;j<data->near_uniq;j++) {
					data->sparsity += data->mult[i]*data->prox[i][j];
				}
			}
			data->sparsity *= (1.0/data->n_params); // divide by the total number of parameters?
			data->sparsity *= exp(sp_d*log(10));

			simple_minimizer(data); // minimize...

			printf("Out of sample\n");
			printf("RMS of J/h: %14.12lf\n", sqrt(gsl_stats_variance(saved_list, 1, data->n_params)));
			
			printf("Estimated KL of data, given inferred: %17.12lf\n", full_kl(data, data->big_list, saved_list)); // many samples, a little correlated			
			printf("RMS of J/h: %14.12lf\n", sqrt(gsl_stats_variance(data->big_list, 1, data->n_params)));
			
			// kl_true=sample_states(data, data->n*10, 100000, saved_list);
			// kl_guess=sample_states(data, data->n*10, 100000, data->big_list);
			// printf("Estimated KL of data, given inferred: %17.12lf\n", kl_guess-kl_true); // many samples, a little correlated
			

		}
		// compute_k_general(data, 0);
		// // printf("Final: %lf\n", data->k);
		// // printf("init=");
		// // pretty(saved_list, data->n_params);
		// // printf("final=");
		// // pretty(data->big_list, data->n_params);
		// // printf("plot, init, final, psym=4 & oplot, findgen(100)-50, findgen(100)-50\n");
		// // printf("Correlation: %14.12lf\n", gsl_stats_correlation(data->big_list, 1, saved_list, 1, data->n_params));
		// //
		// // running=0;
		// // for(i=0;i<data->n_params;i++) {
		// // 	running += (saved_list[i]-data->big_list[i])*(saved_list[i]-data->big_list[i]);
		// // }
		// // printf("RMS: %14.12lf\n", sqrt(running/data->n_params));
		// // printf("Variance: %14.12lf\n", gsl_stats_variance(data->big_list, 1, data->n_params));
		// // printf("plot, init, final, psym=4, xrange=[min(init), max(init)], yrange=[min(init), max(init)]\n");
		//
		//
		// if (saved_list != NULL) {
		// 	final_list=(double *)malloc(data->n_params*sizeof(double));
		// 	for(i=0;i<data->n_params;i++) {
		// 		final_list[i]=data->big_list[i];
		// 		data->big_list[i]=saved_list[i];
		// 	}
		//
		// 	for(i=0;i<data->uniq;i++) {
		// 		free(data->obs[i]);
		// 	}
		// 	free(data->obs);
		// 	data->uniq=0;
		// 	data->obs=(int **)malloc(data->m*sizeof(int *));
		// 	for(i=0;i<data->m;i++) {
		// 		data->obs[i]=(int *)malloc(data->n*sizeof(int));
		// 		for(j=0;j<data->n;j++) {
		// 			data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
		// 		}
		// 		mcmc_sampler(data, i, data->n*atoi(argv[5]), data->big_list);
		// 	}
		//
		// 	printf("Estimated log-L of data, out of sample, given true: %17.12lf\n", sample_states(data, data->n*10, 100000, saved_list)); // many samples, a little correlated
		// 	printf("Estimated log-L of data, out of sample, given inferred: %17.12lf\n", sample_states(data, data->n*10, 100000, final_list)); // many samples, a little correlated
		// }
		
	}
	printf("Clock time: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	exit(1);
}