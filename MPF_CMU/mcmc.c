#include "mpf.h"
#include "stdlib.h"
#define flip(X)  ((X) < 0 ? 1 : -1)
#define flip(X)  ((X) < 0 ? 1 : -1)
#define VAL(a, pos) ((a & (1 << pos)) ? 1.0 : -1.0)

void mcmc_sampler(unsigned long int *config, int iter, all *data) {
	int i, j, pos, count;
	double running, exp_running, rnd, scale;
		
	// quick simulated annealing...
	for(i=0;i<data->n*iter/2;i++) {
		scale=i*1.0/(data->n*iter/2);
		pos=(int)gsl_rng_uniform_int(data->r, data->n); // pick a point randomly
		
		running=0;
		// change in energy function from the proposed flip
		for(j=0;j<data->n;j++) {
			if (pos != j) {
				running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*VAL((*config),j)*data->big_list[data->ij[pos][j]]*scale;
			}
		}
		running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*data->big_list[data->h_offset+pos]*scale;
				
		exp_running=exp(running);
		rnd=gsl_rng_uniform(data->r);
		if (rnd < exp_running/(1+exp_running)) {
			(*config) = ((*config) ^ (1 << pos));
		}
	}

	// quick simulated annealing...
	for(i=0;i<data->n*iter/2;i++) {
		pos=(int)gsl_rng_uniform_int(data->r, data->n); // pick a point randomly
		
		running=0;
		// change in energy function from the proposed flip
		for(j=0;j<data->n;j++) {
			if (pos != j) {
				running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*VAL((*config),j)*data->big_list[data->ij[pos][j]];
			}
		}
		running -= (VAL((*config),pos) - VAL(((*config) ^ (1 << pos)),pos))*data->big_list[data->h_offset+pos];
				
		exp_running=exp(running);
		rnd=gsl_rng_uniform(data->r);
		if (rnd < exp_running/(1+exp_running)) {
			(*config) = ((*config) ^ (1 << pos));
		}
	}
	
}

double full_kl(all *data, double *inferred, double *truth) { // intense, full-enumeration kl calculation... explodes exponentially
	int i, n, ip, jp, sig_ip, sig_jp, count=0;
	double z_inferred=0, z_truth=0, kl=0, max_catch=0;
	double e_inferred, e_truth;
	double t0;

	t0=clock();
	
	n=data->n;
	// first compute the partition function -- we could actually save all the values to memory but it's faster not to; we have to do two loops; one calculates the two partition functions (normalizations) -- the second uses that normalization to compute the probabilities. Beware we are NOT doing checks for underflows/overflows in the exp calculation
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		e_truth=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
		}
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			e_truth += sig_ip*truth[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count]; // data->ij[ip][jp] -- for super-speed, we'll live on the edge
				e_truth += sig_ip*sig_jp*truth[count];
				count++;
			}
		}
		// if (i == 0) { // check to see if we're exploding over our range
		// 	if ((fabs(e_inferred) > 30) || (fabs(e_truth) > 30)) {
		// 		max_catch=-(fabs(e_inferred)/e_inferred)*MAX(e_inferred, e_truth);
		// 	}
		// }
		z_inferred += exp(e_inferred+max_catch);
		z_truth += exp(e_truth+max_catch);
	}
	
	// then compute the kl function
	for(i=0;i<(1 << n);i++) {
		
		e_inferred=0;
		e_truth=0;
		count=0;
		for(ip=0;ip<n;ip++) {
			sig_ip=(i & (1 << ip));
			if (sig_ip) {
				sig_ip=1;
			} else {
				sig_ip=-1;
			}
			e_inferred += sig_ip*inferred[data->h_offset+ip];
			e_truth += sig_ip*truth[data->h_offset+ip];
			for(jp=(ip+1);jp<n;jp++) {
				sig_jp=(i & (1 << jp));
				if (sig_jp) {
					sig_jp=1;
				} else {
					sig_jp=-1;
				}
				e_inferred += sig_ip*sig_jp*inferred[count];
				e_truth += sig_ip*sig_jp*truth[count];
				count++;
			}
		}
		// printf("%i %lf %lf %lf\n", i, kl, e_truth, e_inferred);
		kl += (exp(e_truth+max_catch)/z_truth)*log((exp(e_truth+max_catch)/z_truth)/(exp(e_inferred+max_catch)/z_inferred));
	}
	// printf("Clock time KL: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	
	return kl;
}
