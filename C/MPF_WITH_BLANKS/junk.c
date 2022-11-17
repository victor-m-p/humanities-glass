	// also empirically, we shouldn't try to augment the nns to be equal numbers
	// printf("Near uniq %i\n", data->near_uniq);
	// for(i=0;i<data->uniq;i++) {
	// 	ratio=0;
	// 	for(j=0;j<data->near_uniq;j++){
	// 		ratio += data->prox[i][j];
	// 	}
	// 	if (ratio < data->n) { // if there are missing neighbours for a data point, we need to add some in
	//
	// 		while(ratio < data->n) {
	// 			// create a state
	// 			config=(int *)malloc(data->n*sizeof(int));
	// 			for(ip=0;ip<data->n;ip++) {
	// 				config[ip]=data->obs[i][ip];
	// 			}
	// 			// create a random two step change
	// 			found=0;
	// 			while(found == 0) {
	// 				found=1;
	// 				pos=gsl_rng_uniform_int(data->r, data->n);
	// 				config[pos]=flip(config[pos]);
	// 				pos=gsl_rng_uniform_int(data->r, data->n);
	// 				config[pos]=flip(config[pos]);
	// 				if (compare_simple(config, data->obs[i], data->n) == 0) {
	// 					found=0;
	// 				}
	// 				if (found == 1) {
	// 					for(jp=0;jp<data->near_uniq;jp++) {
	// 						if (compare_simple(config, data->near[jp], data->n) == 0) {
	// 							found=0;
	// 							break;
	// 						}
	// 					}
	// 				}
	// 				if (found == 1) {
	// 					for(jp=0;jp<data->uniq;jp++) {
	// 						if (compare_simple(config, data->obs[jp], data->n) == 0) {
	// 							found=0;
	// 							break;
	// 						}
	// 					}
	// 				}
	// 			}
	// 			printf("New neighbour found (%i) (%i)\n", found, compare_simple(config, data->obs[i], data->n));
	// 			printf("%lu %lu\n", convert(data,data->obs[i]), convert(data,config));
	// 			for(j=0;j<data->n;j++) {
	// 				printf("(%i %i)", data->obs[i][j], config[j]);
	// 			}
	// 			printf("\n");
	// 			//
	// 			data->near_uniq++;
	//
	// 			for(j=0;j<data->uniq;j++) {
	// 				data->prox[j]=(int *)realloc(data->prox[j], data->near_uniq*sizeof(int *));
	// 				data->prox[j][data->near_uniq-1]=0;
	// 			}
	// 			data->prox[i][data->near_uniq-1]=1;
	//
	// 			data->near=(int **)realloc(data->near, data->near_uniq*sizeof(int *));
	// 			data->near[data->near_uniq-1]=config;
	//
	// 			data->near_ok=(int *)realloc(data->near_ok, data->near_uniq*sizeof(int));
	// 			data->near_ok[data->near_uniq-1]=1;
	//
	// 			ratio++;
	// 		}
	// 	}
	// }
	// printf("Near uniq %i\n", data->near_uniq);
	//
	// data->ratio=(double *)malloc(data->uniq*sizeof(double));
	// for(i=0;i<data->uniq;i++) {
	// 	data->ratio[i]=0;
	// 	for(j=0;j<data->near_uniq;j++){
	// 		data->ratio[i] += (double)data->prox[i][j];
	// 	}
	// 	data->ratio[i]=((double)data->n/data->ratio[i]);
	// }
	// for(i=0;i<data->uniq;i++) {
	// 	ratio=0;
	// 	for(j=0;j<data->near_uniq;j++) {
	// 		ratio += data->prox[i][j];
	// 	}
	// 	printf("Ratio: %i\n", ratio);
	// }

  	// empirically, we shouldn't remove data states from the neighbour set
  	// what if the number of neighbours differs?
		// for(j=0;j<data->uniq;j++) { // cycle through the data to see if there's a hit...
		// 	if (compare_simple(data->near[i], data->obs[j], data->n) == 0) {
		// 		printf("Found overlap %s %s \n", itoa(convert(data, data->obs[j]),2), itoa(convert(data, data->near[i]),2));
		// 		// data->near_ok[i]=0; // if there is, blank out near_ok
		// 		break;
		// 	}
		// }

compute_k(data, 0);
printf("Base solution: %lf\n", data->k);

double par2[15] = { -0.07097891397915604,0.09037968768574217,0.03953863835261531,0.004395573116284663,0.04214144415742476,0.04975525546955612,0.16313034585464473,0.0984049625571094,0.06165209798185289,0.08770363053852132,0.16436445032747424,0.050891948884650234,0.12236974326387334,0.237981027967441,0.19452103400974274 };
data->big_list = par2; 
compute_k(data, 0);		
printf("Eddie solution: %lf\n", data->k);
printf("Fields\n");
for(i=data->h_offset;i<data->n_params;i++) {
	printf("%lf ", data->big_list[i]);
}
printf("\n");

printf("Pairs\n");
for(i=0;i<data->h_offset;i++) {
	printf("%lf ", data->big_list[i]);
}
printf("\n\n");

simple_minimizer(data);

printf("Fields\n");
for(i=data->h_offset;i<data->n_params;i++) {
	printf("%lf ", data->big_list[i]);
}
printf("\n");

printf("Pairs\n");
for(i=0;i<data->h_offset;i++) {
	printf("%lf ", data->big_list[i]);
}
printf("\n");
