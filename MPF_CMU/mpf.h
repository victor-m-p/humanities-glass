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
	int *config_base; // base configuration (-1, +1, 0)
	double mult; // multiplicity of base config (including blanks)


	int n_blanks; // number of blanks
	int n_config; // 2^n_blanks
	int *blanks; // location of blanks

	// for the below, this is filled in *even* if there are no blanks
	
	unsigned long int *config; // all the configurations filled in and represented as an unsigned long for speed; number of configs is (1 << n_blanks) -- this is written once, then 
	double *mult_sim; // list of simulated conditional multiplicity configurations
	double **mult_sim_pointer; // pointers to the multi_sim in the all list
	int **prox;
} sample;

typedef struct {
	int m;	// number of samples
	int uniq; // number of unique samples
	int n; 	// number of nodes
	int max_config; // max number of blank configs 2^max_blanks
	int max_blanks; // max number of blanks
	
	int near_uniq; // total number of neighbours -- this will be constant for everyone
	int n_prox; // number of neighbours of any point -- this will be constant for everyone
	unsigned long int *near; // the list of neighbours, represented as unsigned long int for speed.
	
	sample **obs_raw; // sample that is read in; this will be sorted and duplicate-filtered
	sample **obs; // the observations that we actually work with
	
	double log_sparsity;
	double sparsity;
	
	int n_all;
	
	int n_params;	
	double *big_list;
	double *old_list;
	
	double *nei;
	double *ei;
	
	double k; 
	double *dk;
	
	int **ij;
	int h_offset;
	
	gsl_rng *r;
} all;

gsl_rng *rng();

all *new_data();
void read_data(char *filename, all *data);
void process_obs_raw(all *data);
void init_params(all *data);
void create_near(all *data, int n_step);
void update_mult_sim(all *data);

void compute_k_general(all *data, int do_derivs);
void simple_minimizer(all *data);

void print_vec(unsigned long a);
unsigned long int convert(int *list);