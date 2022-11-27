#include "mpf.h"
// mpf -l [filename] [logsparsity] [NN] // load in data, simulate
// mpf -g [filename] [n_obs] [n_nodes] [beta] // simulate data, save both parameters and 

int main (int argc, char *argv[]) {
	double t0, old, ep=1e-2, acc;
	all *data;
	int i, j;
	
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
		if (argv[1][1] == 's') {
			
		}
		if (argv[1][1] == 'o') {
			
		}
	}
	printf("Clock time: %14.12lf seconds.\n", (clock() - t0)/CLOCKS_PER_SEC);
	exit(1);
}