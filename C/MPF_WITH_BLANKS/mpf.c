#include "mpf.h"
#include "stdlib.h"
#define flip(X)  ((X) < 0 ? 1 : -1)
#define MIN(a,b) ((a)<(b)?(a):(b))

char* itoa(unsigned long val, int base){
	static char buf[64] = {0};
	unsigned long val_temp=val;
	unsigned long i = 64;
	
	for(; val_temp && i ; --i, val_temp /= base)
	
		buf[i] = "0123456789abcdef"[val_temp % base];
	
	return &buf[i+1];
	
}

int one_step(int *a, int *b, int len) { // checks if a and b differ by one step
	int i, diff=0;
	
	for(i=0;i<len;i++) {
		if (a[i] != b[i]) {
			diff++;
		}
		if (diff > 1) {
			return 0;
		}
	}
	return diff;
}

int two_step(int *a, int *b, int len) { // checks if a and b differ by two steps
	int i, diff=0;
	
	for(i=0;i<len;i++) {
		if (a[i] != b[i]) {
			diff++;
		}
		if (diff > 2) {
			return 0;
		}
	}
	if (diff == 2) {
		return 1;
	} else {
		return diff;		
	}
}

int three_step(int *a, int *b, int len) { // checks if a and b differ by two steps
	int i, diff=0;
	
	for(i=0;i<len;i++) {
		if (a[i] != b[i]) {
			diff++;
		}
		if (diff > 3) {
			return 0;
		}
	}
	if (diff > 0) {
		return 1;
	} else {
		return 0;		
	}
}

int four_step(int *a, int *b, int len) { // checks if a and b differ by two steps
	int i, diff=0;
	
	for(i=0;i<len;i++) {
		if (a[i] != b[i]) {
			diff++;
		}
		if (diff > 4) {
			return 0;
		}
	}
	if (diff > 0) {
		return 1;
	} else {
		return 0;		
	}
}

int global_length; // this is crazy, but we need a global variable because BSD, GNU, and Microsoft could not agree on a consensus library definition of qsort_r

void pretty(double *list, int n) {
	int i;
	
	printf("[");
	for(i=0;i<(n-1);i++) {
		printf("%14.12lf, ", list[i]);
	}
	printf("%lf]\n", list[n-1]);
}

int compare_states(const void* a, const void* b) {
	int i, **arg1, **arg2;
 	
	arg1=(int **)a;
 	arg2=(int **)b;
	
 	for(i=0;i<global_length;i++) {
 		if (arg1[0][i] > arg2[0][i]) return -1;
 		if (arg1[0][i] < arg2[0][i]) return 1;		
 	}
    return 0;
}

int compare_simple(int *a, int *b, int len) {
	int i;
	
 	for(i=0;i<len;i++) {
 		if (a[i] > b[i]) return 1;
 		if (a[i] < b[i]) return -1;		
 	}
    return 0;	
}

void create_near(samples *data, int n_step) { // creates nearest neighbours, removes duplicates
	int i, j, jp, jpp, jppp, k, count, pos, num_near, **near_temp, count_uniq;
		
	count=0;
	// the basic case (check this against our code)
	if (n_step == 1) {		
		num_near=data->uniq*data->n; // num near = uniq * n (makes sense)
		data->near=(int **)malloc(num_near*sizeof(int *)); // allocating
		for(i=0;i<data->uniq;i++) { // loop over uniq
			for(j=0;j<data->n;j++) { // loop over n 
				data->near[count]=(int *)malloc(data->n*sizeof(int));
				for(k=0;k<data->n;k++) { // loop over n 
					data->near[count][k]=data->obs[i][k]; 
				}
				data->near[count][j]=flip(data->near[count][j]);
				count++;
			}
		}
	} else {
		if (n_step == 2) {
			num_near=data->uniq*(data->n+data->n*data->n);	
			data->near=(int **)malloc(num_near*sizeof(int *));	
			count=0;
			for(i=0;i<data->uniq;i++) {
				for(j=0;j<data->n;j++) {
					data->near[count]=(int *)malloc(data->n*sizeof(int));
					for(k=0;k<data->n;k++) {
						data->near[count][k]=data->obs[i][k];
					}
					data->near[count][j]=flip(data->near[count][j]);
					count++;
				}

				for(j=0;j<data->n;j++) {
					for(jp=0;jp<data->n;jp++) {
						data->near[count]=(int *)malloc(data->n*sizeof(int));
						for(k=0;k<data->n;k++) {
							data->near[count][k]=data->obs[i][k];
						}
						data->near[count][j]=flip(data->near[count][j]);
						data->near[count][jp]=flip(data->near[count][jp]);
						count++;												
					}
				}	
			}
		} else {
			if (n_step == 3) {
				num_near=data->uniq*(data->n+data->n*data->n+data->n*data->n*data->n);	
				data->near=(int **)malloc(num_near*sizeof(int *));	
				count=0;
				for(i=0;i<data->uniq;i++) {
					for(j=0;j<data->n;j++) {
						data->near[count]=(int *)malloc(data->n*sizeof(int));
						for(k=0;k<data->n;k++) {
							data->near[count][k]=data->obs[i][k];
						}
						data->near[count][j]=flip(data->near[count][j]);
						count++;
					}

					for(j=0;j<data->n;j++) { // add a sprinkling of two-step
						for(jp=0;jp<data->n;jp++) {
							data->near[count]=(int *)malloc(data->n*sizeof(int));
							for(k=0;k<data->n;k++) {
								data->near[count][k]=data->obs[i][k];
							}
							data->near[count][j]=flip(data->near[count][j]);
							data->near[count][jp]=flip(data->near[count][jp]);
							count++;												
						}
					}
					
					for(j=0;j<data->n;j++) { // add a sprinkling of two-step
						for(jp=0;jp<data->n;jp++) {
							for(jpp=0;jpp<data->n;jpp++) {
								data->near[count]=(int *)malloc(data->n*sizeof(int));
								for(k=0;k<data->n;k++) {
									data->near[count][k]=data->obs[i][k];
								}
								data->near[count][j]=flip(data->near[count][j]);
								data->near[count][jp]=flip(data->near[count][jp]);
								data->near[count][jpp]=flip(data->near[count][jpp]);
								count++;												
							}
						}
					}
				}
			} else {
				if (n_step == 4) {
					num_near=data->uniq*(data->n+data->n*data->n+data->n*data->n*data->n+data->n*data->n*data->n*data->n);	
					data->near=(int **)malloc(num_near*sizeof(int *));	
					count=0;
					for(i=0;i<data->uniq;i++) {
						for(j=0;j<data->n;j++) {
							data->near[count]=(int *)malloc(data->n*sizeof(int));
							for(k=0;k<data->n;k++) {
								data->near[count][k]=data->obs[i][k];
							}
							data->near[count][j]=flip(data->near[count][j]);
							count++;
						}

						for(j=0;j<data->n;j++) { // add a sprinkling of two-step
							for(jp=0;jp<data->n;jp++) {
								data->near[count]=(int *)malloc(data->n*sizeof(int));
								for(k=0;k<data->n;k++) {
									data->near[count][k]=data->obs[i][k];
								}
								data->near[count][j]=flip(data->near[count][j]);
								data->near[count][jp]=flip(data->near[count][jp]);
								count++;												
							}
						}
					
						for(j=0;j<data->n;j++) { // add a sprinkling of two-step
							for(jp=0;jp<data->n;jp++) {
								for(jpp=0;jpp<data->n;jpp++) {
									data->near[count]=(int *)malloc(data->n*sizeof(int));
									for(k=0;k<data->n;k++) {
										data->near[count][k]=data->obs[i][k];
									}
									data->near[count][j]=flip(data->near[count][j]);
									data->near[count][jp]=flip(data->near[count][jp]);
									data->near[count][jpp]=flip(data->near[count][jpp]);
									count++;												
								}
							}
						}

						for(j=0;j<data->n;j++) { // add a sprinkling of two-step
							for(jp=0;jp<data->n;jp++) {
								for(jpp=0;jpp<data->n;jpp++) {
									for(jppp=0;jppp<data->n;jppp++) {
										data->near[count]=(int *)malloc(data->n*sizeof(int));
										for(k=0;k<data->n;k++) {
											data->near[count][k]=data->obs[i][k];
										}
										data->near[count][j]=flip(data->near[count][j]);
										data->near[count][jp]=flip(data->near[count][jp]);
										data->near[count][jpp]=flip(data->near[count][jpp]);
										data->near[count][jppp]=flip(data->near[count][jppp]);
										count++;												
										
									}
								}
							}
						}

					}
				}
				
			}
		}
	}
	qsort(data->near, num_near, sizeof(int **), compare_states);

	count_uniq=1;
	i=1;
	while(i<num_near) {
		if (compare_simple(data->near[i], data->near[i-1], data->n) != 0) {
			count_uniq++;
		}
		i++;
	}
	near_temp=(int **)malloc(count_uniq*sizeof(int *));
	data->near_uniq=count_uniq;

	i=0;
	pos=1;
	near_temp[0]=data->near[0];
	while(pos<num_near) { 
		if (compare_simple(data->near[pos], data->near[pos-1], data->n) != 0) { // if the current one is different from the previous one then...
			i++; // increment the counter...
			near_temp[i]=data->near[pos]; // save the new one...
		}
		pos++;
	}
	free(data->near); // so fucking edgy
	data->near=near_temp;

	data->near_ok=(int *)malloc(data->near_uniq*sizeof(int *));
}

void resimulate_missing(samples *data) { // takes the raw data, recreates the system
	int i, j;
	
	// first, delete the current observation set...
	for(i=0;i<data->uniq;i++) {
		free(data->obs[i]);
	}
	free(data->obs);
	
	// then, create it again
	data->obs=(int **)malloc(data->m*sizeof(int *));
	for(i=0;i<data->m;i++) {
		data->obs[i]=(int *)malloc(data->n*sizeof(int));
		for(j=0;j<data->n;j++) {
			if (data->obs_raw[i][j] == 2) {
				data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1); 
			} else {
				data->obs[i][j]=data->obs_raw[i][j];				
			}
		}
		if (data->n_blanks[i] > 0) {
			mcmc_sampler_partial(data, i, data->n_blanks[i]*10, data->big_list); // iterate ten times to get the sample
		}
	}
}

void create_neighbours(samples *data, int n_steps) {
	int i, j, ip, jp, found, count_uniq, pos, multiplicity, ratio;
	int *config;
	
	if (data->prox != NULL) { // not sure what prox is 
		for(i=0;i<data->near_uniq;i++) { // we need to know what near_uniq is 
			free(data->near[i]); // and what near is
		}
		free(data->near);
		free(data->near_ok);
		for(i=0;i<data->uniq;i++) {
			free(data->prox[i]);
		}
		free(data->prox);
	}

					// one_step just returns whether it is one step away I think
	create_near(data, n_steps); // need to create all the n_step NNs
	
	for(i=0;i<data->near_uniq;i++) { // for each of the neighbour states...
		data->near_ok[i]=1; // always 1, so what is the point?
	}
	
	data->prox=(int **)malloc(data->uniq*sizeof(int *));
	if (n_steps == 1) {
		for(i=0;i<data->uniq;i++) {
			data->prox[i]=(int *)malloc(data->near_uniq*sizeof(int));
			for(j=0;j<data->near_uniq;j++) {
				if (data->near_ok[j] == 1) {
					// one_step just returns whether it is one step away I think
					data->prox[i][j]=one_step(data->obs[i], data->near[j], data->n);
				} else {
					data->prox[i][j]=0;
				}	
			}
		}		
	} else {
		if (n_steps == 2) {
			for(i=0;i<data->uniq;i++) {
				data->prox[i]=(int *)malloc(data->near_uniq*sizeof(int));
				for(j=0;j<data->near_uniq;j++) {
					if (data->near_ok[j] == 1) {
						data->prox[i][j]=two_step(data->obs[i], data->near[j], data->n);
					} else {
						data->prox[i][j]=0;
					}	
				}
			}					
		}
		if (n_steps == 3) {
			for(i=0;i<data->uniq;i++) {
				data->prox[i]=(int *)malloc(data->near_uniq*sizeof(int));
				for(j=0;j<data->near_uniq;j++) {
					if (data->near_ok[j] == 1) {
						data->prox[i][j]=three_step(data->obs[i], data->near[j], data->n);
					} else {
						data->prox[i][j]=0;
					}	
				}
			}					
		}
		if (n_steps == 4) {
			for(i=0;i<data->uniq;i++) {
				data->prox[i]=(int *)malloc(data->near_uniq*sizeof(int));
				for(j=0;j<data->near_uniq;j++) {
					if (data->near_ok[j] == 1) {
						data->prox[i][j]=four_step(data->obs[i], data->near[j], data->n);
					} else {
						data->prox[i][j]=0;
					}	
				}
			}					
		}
		
	}
		
	if (data->nei != NULL) {
		free(data->nei);
	}
	data->nei=(double *)malloc(data->near_uniq*sizeof(double));
	
}

void sort_and_filter(samples *data) { 
	// this is going to slim down the obs states...
	// then it is going to recompute the nearest neighbours...
	int i, j, ip, jp, found, count_uniq, pos, multiplicity, ratio;
	int **obs_temp, *config;
	
	global_length=data->n;
	
	qsort(data->obs, data->m, sizeof(int **), compare_states);
	count_uniq=1;
	i=1;
	while(i<(data->m)) {
		if (compare_simple(data->obs[i], data->obs[i-1], data->n) != 0) {
			count_uniq++;
		}
		i++;
	}
			
	obs_temp=(int **)malloc(count_uniq*sizeof(int *));
	data->uniq=count_uniq;
	
	if (data->mult != NULL) {
		free(data->mult);
	}
	data->mult=(int *)malloc(count_uniq*sizeof(int));
		
	i=0;
	pos=1;
	multiplicity=1;
	obs_temp[0]=data->obs[0];
	while(pos<data->m) { // need to FIX THIS TKTK ??? -- what does this mean??
		if (compare_simple(data->obs[pos], data->obs[pos-1], data->n) != 0) { // if the current one is different from the previous one then...
			i++; // increment the counter...
			obs_temp[i]=data->obs[pos]; // save the new one...
			data->mult[i-1]=multiplicity; // save the old multiplicity...
			multiplicity=0;
		}
		pos++;
		multiplicity++;
	}
	data->mult[count_uniq-1]=multiplicity;
		
	free(data->obs); // so fucking edgy
	data->obs=obs_temp;
	
	if (data->ei != NULL) {
		free(data->ei);
	}
	data->ei=(double *)malloc(data->uniq*sizeof(double));	
}

samples *new_data() {
	FILE *fn; 
	unsigned long r_seed;

	samples *data;
	
	data=(samples *)malloc(sizeof(samples));
	data->big_list=NULL;
	data->ei=NULL;
	data->nei=NULL;
	data->dk=NULL;
	
	data->ij=NULL;
	
	data->obs_raw=NULL;
	data->obs=NULL;
	data->n_blanks=NULL;
	data->blanks=NULL;
	data->mult=NULL;
	data->prox=NULL;
	
	data->uniq=0;
	
	fn = fopen("/dev/urandom", "rb");
	
	if (fread(&r_seed, sizeof(unsigned long), 1, fn) != 1) {
		/* Failed!--use time instead; beware, could sync with other instances */
		printf("Warning: urandom read fail; using system clock\n");
		r_seed=(unsigned long)time(NULL);
	}
	fclose(fn);
	
	data->r=gsl_rng_alloc(gsl_rng_taus);
	gsl_rng_set(data->r, r_seed);
	
	return data;
}

void delete_data(samples *data) {
	int i,j;
	
	if (data->obs != NULL) {
		for(i=0;i<data->m;i++) {
			free(data->obs[i]);
		}
		free(data->obs);
	}
	
	if (data->big_list != NULL) {
		free(data->big_list);
	}

	if (data->ij != NULL) {
		for(i=0;i<data->n;i++) {
			free(data->ij[i]);
		}
		free(data->ij);
	}

	if (data->ei != NULL) {
		free(data->ei);
	}
	if (data->dk != NULL) {
		free(data->dk);
	}
		
	if (data->n_blanks != NULL) {
		for(i=0;i<data->m;i++) {
			for(j=0;j<data->n_blanks[i];j++) {
				free(data->blanks[i]);
			}
		}
		free(data->blanks);
		free(data->n_blanks);
	}
	free(data);
}

void blank_system(samples *data, int m, int n) {
	int i, j;
	
	data->m=m;
	data->n=n;
	data->obs=(int **)malloc(m*sizeof(int *));
	data->obs_raw=(int **)malloc(m*sizeof(int *));
	global_length=data->n;

	data->n_blanks=(int *)malloc(m*sizeof(int));
	data->blanks=(int **)malloc(m*sizeof(int *));

	for(i=0;i<data->m;i++) {
		data->obs[i]=(int *)malloc(n*sizeof(int));
		data->obs_raw[i]=(int *)malloc(n*sizeof(int));
		data->n_blanks[i]=0;
		data->blanks[i]=NULL;
		for(j=0;j<data->n;j++) {
			data->obs[i][j]=(2*gsl_rng_uniform_int(data->r, 2)-1);
			data->obs_raw[i][j]=data->obs[i][j];
		}
	}
	data->h_offset=data->n*(data->n-1)/2;
	data->n_params=data->n*(data->n-1)/2+data->n;
	
}

void load_data(char *filename, samples *data) {
	int i, j, count, m, n;
	FILE *f_in;
	char c;
	
	f_in=fopen(filename, "r");
	
	fscanf(f_in, "%i\n", &(data->m)); // number of samples
	fscanf(f_in, "%i\n", &(data->n)); // number of nodes
	m=data->m;
	global_length=data->n;
	n=data->n;
	
	data->obs=(int **)malloc(m*sizeof(int *));
	data->obs_raw=(int **)malloc(m*sizeof(int *));

	data->n_blanks=(int *)malloc(m*sizeof(int));
	data->blanks=(int **)malloc(m*sizeof(int *));
	
	for(i=0;i<data->m;i++) {
		data->obs[i]=(int *)malloc(n*sizeof(int));
		data->obs_raw[i]=(int *)malloc(n*sizeof(int));
		data->n_blanks[i]=0;
		data->blanks[i]=NULL;
		for(j=0;j<n;j++) {
			data->obs[i][j]=-100;
			fscanf(f_in, "%c", &c);
			if (c == '0') {
				data->obs_raw[i][j]=-1;
				data->obs[i][j]=-1;
			}
			if (c == '1') {
				data->obs_raw[i][j]=1;
				data->obs[i][j]=1;
			}
			if (c == 'X') {
				data->obs_raw[i][j]=2;
				data->obs[i][j]=-1; // DEFAULT FOR NOW -- THIS WILL GET FLIPPED BY THE SIMULATOR
				data->n_blanks[i]++;
			}
			if (data->obs[i][j] == -100) {
				printf("Bad entry for node %i of obs %i; entry was %c\n", j, i, c);
			}
		}
		
		if (data->n_blanks[i] > 0) { // now register all the missing data
			data->blanks[i]=(int *)malloc(data->n_blanks[i]*sizeof(int));
			count=0;
			for(j=0;j<n;j++) {
				if (data->obs_raw[i][j] == 2) {
					data->blanks[i][count]=j;
					count++;					
				}
			}
		}
		
		fscanf(f_in, "%c", &c);
		if ((c != '\n') && (i < (m-1))) {
			printf("Expected an end of line, didn't get one\n");
		}
	}
	fclose(f_in);
	data->h_offset=data->n*(data->n-1)/2;
	data->n_params=data->n*(data->n-1)/2+data->n;
}

void init_params(samples *data, int fit_data) {
	int i, j, d, count;
	double running;
	gsl_rng *r;
	
	if (data->big_list == NULL) {
		data->big_list=(double *)malloc(data->n_params*sizeof(double));
		data->dk=(double *)malloc(data->n_params*sizeof(double));
		r=data->r;
		
		for(i=0;i<data->m;i++) {
			if (data->n_blanks[i] > 0) {
				for(j=0;j<data->n_blanks[i];j++) {
					data->obs[i][data->blanks[i][j]] = (2*gsl_rng_uniform_int(r, 2)-1); // randomly set the initial data to -1, 1
				}
			}
		}

		data->ij=(int **)malloc(data->n*sizeof(int *));
		for(i=0;i<data->n;i++) {
			data->ij[i]=(int *)malloc(data->n*sizeof(int));
		}
		count=0;
		for(i=0;i<(data->n-1);i++) {
			for(j=(i+1);j<data->n;j++) {
				data->ij[i][j]=count;
				data->ij[j][i]=count;
				count++;
			}
		}		
	}

	// guess the initial correlations as the J_ijs etc.
	if (fit_data == 1) {
		for(i=0;i<data->n_params;i++) {
			data->big_list[i]=gsl_ran_gaussian(data->r, 1.0)/100.0; // small symmetry-breaking values
		}
		// for(i=0;i<data->n;i++) {
		// 	data->big_list[data->h_offset+i] = 0; // set h to the mean value...
		// 	for(d=0;d<data->m;d++) {
		// 		data->big_list[data->h_offset+i] += data->obs[d][i];
		// 	}
		// 	data->big_list[data->h_offset+i] = data->big_list[data->h_offset+i]/data->m + gsl_rng_uniform(data->r)/10.0; // + gsl_rng_uniform(r)/10.0;
		// }
		// for(i=0;i<(data->n-1);i++) {
		// 	for(j=(i+1);j<data->n;j++) {
		// 		running=0;
		// 		for(d=0;d<data->m;d++) {
		// 			running += (data->obs[d][i]-data->big_list[data->h_offset+i])*(data->obs[d][j]-data->big_list[data->h_offset+j]);
		// 		}
		// 		data->big_list[data->ij[i][j]] = running/data->m + gsl_rng_uniform(data->r)/10.0; // + gsl_rng_uniform(r)/10.0;
		// 	}
		// }
	} else {
		for(i=0;i<data->n_params;i++) {
			data->big_list[i]=gsl_ran_gaussian(data->r, 1.0); // initiatize for tests
		}
	}

}

void compute_k_general(samples *data, int do_derivs) { // 
	int d, a, i, j, ip, jp, n, count, term;
	int **ij, *config, *config1, *config2, val;
	double **obs;
	double max_val, *ei, energy, running, multiplier;
		
	ij=data->ij; // save typing
		
	if (do_derivs == 1) {
		for(i=0;i<data->n_params;i++) {
			data->dk[i]=0;
		}		
	}
	
	for(d=0;d<data->uniq;d++) { // for each unique datapoint...
		config=data->obs[d];
		data->ei[d]=0;
		count=0;
		for(i=0;i<data->n;i++) {
			for(j=(i+1);j<data->n;j++) {
				data->ei[d] += (double)config[i]*(double)config[j]*data->big_list[count];
				count++;
			}
			data->ei[d] += (double)config[i]*data->big_list[data->h_offset+i]; // local fields
		}
		data->ei[d] *= -1; // defined as the negative value in Jascha paper
	}

	for(d=0;d<data->near_uniq;d++) { // for each unique nearest neighbour...
		if (data->near_ok[d] == 1) {
			config=data->near[d];
			data->nei[d]=0;
			count=0;
			for(i=0;i<data->n;i++) {
				for(j=(i+1);j<data->n;j++) {
					data->nei[d] += (double)config[i]*(double)config[j]*data->big_list[count];
					count++;
				}
				data->nei[d] += (double)config[i]*data->big_list[data->h_offset+i]; // local fields
			}	
			data->nei[d] *= -1; // defined as the negative value in Jascha paper
		}	
	}

	max_val=-1e300;
	for(d=0;d<data->uniq;d++) {
		for(n=0;n<data->near_uniq;n++) {
			if ((data->ei[d]-data->nei[n]) > max_val) {
				max_val=(data->ei[d]-data->nei[n]);
			}
		}
	}	
	
	data->k=0;
	running=0;
	max_val=max_val/2.0;
	for(d=0;d<data->uniq;d++) {
		config1=data->obs[d];
		for(n=0;n<data->near_uniq;n++) { // edit this to restrict the number of NNs considered for each datapoint
			if (data->prox[d][n] == 1) {
				config2=data->near[n];
				multiplier=data->mult[d]*exp(0.5*(data->ei[d]-data->nei[n])-max_val); // NOTE RATIO HERE... ratio trick doesn't work data->ratio[d]*
				data->k += multiplier;
			
				// hard part -- do the derivatives
				if (do_derivs == 1) {
					count=0;
					for(ip=0;ip<(data->n);ip++) {
						for(jp=(ip+1);jp<data->n;jp++) {
							data->dk[count] += -1*(config1[ip]*config1[jp]-config2[ip]*config2[jp])*multiplier/2.0;  
							// defined as the negative value in Jascha paper -- BUT: note that Jascha was off by a factor of 1/2, Eddie fixed it
							count++;
						}
					}
					for(ip=0;ip<(data->n);ip++) {
						data->dk[data->h_offset+ip] += -1*(config1[ip]-config2[ip])*multiplier/2.0;  // defined as the negative value in Jascha paper
					}
				}				
			}
		}
	}
	data->k = (data->k)*exp(max_val); // /data->m
		
	for(i=0;i<data->n_params;i++) {
		data->k += data->sparsity*data->big_list[i]*data->big_list[i]/2; // put in a sparse prior...
	}
	
	if (do_derivs == 1) {
		for(i=0;i<data->n_params;i++) {
			data->dk[i]=(data->dk[i])*exp(max_val) + data->sparsity*data->big_list[i]; // don't forget that this also impacts the derivatives!
			// /data->m
		}
	}
}


// GSL FUNCTIONS DEFINED
double k_function(const gsl_vector *v, void *params) {
	int i;
	samples *data;
	
	data=(samples *)params;
	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(v, i);
	}
	compute_k_general(data, 0);
	
	return data->k;
}

void dk_function(const gsl_vector *v, void *params, gsl_vector *df) {
	int i;
	samples *data;
	
	data=(samples *)params;
	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(v, i);
	}
	compute_k_general(data, 1);
	
	for(i=0;i<data->n_params;i++) {
		gsl_vector_set(df, i, data->dk[i]);
	}
}
	
void kdk_function(const gsl_vector *x, void *params, double *f, gsl_vector *df) {
    *f = k_function(x, params);
    dk_function(x, params, df);
}

void simple_minimizer(samples *data) {
    size_t iter = 0;
	double prev;
    int i, status;

    const gsl_multimin_fdfminimizer_type *T;
    gsl_multimin_fdfminimizer *s;

	gsl_vector *x;
	gsl_multimin_function_fdf k_func;

	// set up the system
	k_func.n = data->n_params;  /* number of function components */
	k_func.f = &k_function;
	k_func.df = &dk_function;
	k_func.fdf = &kdk_function;
	k_func.params = (void *)data;
	
	x = gsl_vector_alloc(data->n_params);
	for(i=0;i<data->n_params;i++) {
		gsl_vector_set(x, i, data->big_list[i]);
	}
	T = gsl_multimin_fdfminimizer_conjugate_fr;
	T = gsl_multimin_fdfminimizer_vector_bfgs2;
	s = gsl_multimin_fdfminimizer_alloc(T, data->n_params);
	
	compute_k_general(data, 1);

	// printf("Initial %lf\n", data->k);
	// for(i=0;i<data->n_params;i++) {
	// 	printf("%lf ", data->big_list[i]);
	// }
	// printf("\n");

	gsl_multimin_fdfminimizer_set(s, &k_func, x, 0.1, 1e-4);

	prev=1e300;
	do {
		iter++;
		status = gsl_multimin_fdfminimizer_iterate(s);
		// here is where we would re-simulate the missing data

		status = gsl_multimin_test_gradient(s->gradient, 1e-4);
		
		// printf ("%i %li (%lf) : ", status, iter, s->f);
		// for(i=0;i<data->n_params;i++) {
		// 	printf("%lf ", gsl_vector_get (s->x, i));
		// }
		// printf("\n");
		//
		// if (status == GSL_SUCCESS) {
		// 	printf ("Minimum found at iter (%li): %lf\n", iter, s->f);
		// }
		
		if (fabs(prev-s->f) < 1e-8) {
			break;
		}
		prev=s->f;
	} while (status == GSL_CONTINUE && iter < 5000);

	compute_k_general(data, 0);
	for(i=0;i<data->n_params;i++) {
		data->big_list[i]=gsl_vector_get(s->x, i);
	}
	
	gsl_multimin_fdfminimizer_free(s);
	gsl_vector_free(x);
}








