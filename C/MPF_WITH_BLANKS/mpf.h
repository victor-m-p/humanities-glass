#include <stdio.h> 
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <unistd.h> // we use this to get process ID and help randomness

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_statistics_double.h>

#include <gsl/gsl_sf.h>
//#include <gsl/gsl_errno.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_integration.h>

#define BIG 1e6
#define BIGI 1e-6
#define EPSILON 1e-16

typedef struct {
	int m;	// number of samples
	int n; 	// number of nodes
	int **obs_raw; // observations
	int **obs; // simulated observations
	int uniq; // number of unique simulated observations
	int upsample; // inflation factor for simulator
	int *mult; // duplicate counts
	int **near;
	int near_uniq;
	int *near_ok;
	int **prox;
	double sparsity; 
	int *n_blanks; // how many blanks
	int **blanks; // where are the missing data

	int n_params;
	
	double *big_list;
	double *big_list_compare;
	
	double *ei;
	double *nei;
	
	double k; 
	double *dk;
	double de;
	
	int **ij;
	int h_offset;
	
	gsl_rng *r;
} samples;

typedef struct {
	int n;	// number of states
	double *p; // list of probabilities
	double norm; // overall normalization
	double holder; // useful holder
	gsl_ran_discrete_t *pre_proc; // the pre-processor for fast number generation
} prob;

typedef struct {
	int n;	// number of states for process I
	int m; // number of states for process II
	
	double **p; // list of probabilities
	double holder; // useful holder
	double norm; // overall normalization
	gsl_ran_discrete_t *pre_proc; // the pre-processor for fast number generation
} j_prob;

gsl_rng *rng();

samples *new_data();
void load_data(char *filename, samples *data);
void init_params(samples *data, int fit_data);
void sort_and_filter(samples *data);
void create_neighbours(samples *data, int n_steps);
double find_max(double *sp, double *logl, int n);
	
void compute_k(samples *data, int do_derivs);
void compute_k_general(samples *data, int do_derivs);

void simple_minimizer(samples *data);
unsigned long convert(samples *data, int *config);
	
void delete_data(samples *data);

void blank_system(samples *data, int m, int n);
void mcmc_sampler(samples *data, int loc, int iter, double *big_list);
void mcmc_sampler_partial(samples *data, int loc, int iter, double *big_list);
double sample_states(samples *data, int iter, int n_samp, double *big_list);
double compare_jij(samples *data, int iter, int n_samp, double *big_list, double *big_list_2);
double kl_est(samples *data, double *big_list, double *big_list_2);
double full_kl(samples *data, double *inferred, double *truth);
	
void pretty(double *list, int n);

double entropy_nsb(prob *p);