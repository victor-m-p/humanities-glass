#include "mpf.h"
// mpf -l [filename] [logsparsity] [NN] // load in data, simulate
// mpf -s n_obs n_nodes beta iter nn // simulate a random system with n_obs, n_nodes, and beta (scaling parameter for J_ij and h_i)
// mpf -o n_obs n_nodes beta iter sparsity nn // simulate a random system with n_obs, n_nodes, and beta (scaling parameter for J_ij and h_i)

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

			// testing fix here...
			// data->sparsity=0;
			// for(i=0;i<data->n_params;i++) {
			// 	data->big_list[i]=0.2;
			// }
			//
			// compute_k_general(data, 1);
			// for(i=0;i<data->n_params;i++) {
			// 	printf("%lf ", data->dk[i]);
			// }
			// printf("\n");
			//
			// old=data->k;
			// acc=0;
			// ep=atof(argv[5]);
			// for(i=0;i<data->n_params;i++) {
			// 	data->big_list[i] += ep;
			// 	compute_k_general(data, 0);
			// 	old=data->k;
			// 	data->big_list[i] -= 2*ep;
			// 	compute_k_general(data, 0);
			// 	printf("%.5le ", (data->dk[i]-((old-data->k)/(2*ep)))/fabs(data->dk[i]));
			// 	acc += fabs(data->dk[i]-((old-data->k)/(2*ep)))/fabs(data->dk[i]);
			// 	data->big_list[i] += ep;
			// }
			// printf("\n");
			// printf("MEAN ACC: %le\n", acc/data->n_params);
									
			simple_minimizer(data);
			//
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