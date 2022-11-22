#include <mpf.h>
#include <stdio.h>
#include <gsl/gsl_multifit.h>

double find_max(double *sp, double *logl, int n) {
	int i;
	gsl_matrix *X, *cov;
	gsl_vector *y, *w, *c;
	double max_loc;

	X = gsl_matrix_alloc (n, 3);
	y = gsl_vector_alloc (n);
	w = gsl_vector_alloc (n);

	c = gsl_vector_alloc (3);
	cov = gsl_matrix_alloc (3, 3);

	for (i=0; i<n; i++) {
		gsl_matrix_set (X, i, 0, 1.0);
		gsl_matrix_set (X, i, 1, sp[i]);
		gsl_matrix_set (X, i, 2, sp[i]*sp[i]);

		gsl_vector_set (y, i, logl[i]);
		gsl_vector_set (w, i, 0.1);
	}

	gsl_multifit_linear_workspace *work = gsl_multifit_linear_alloc (n, 3);
	gsl_multifit_wlinear (X, w, y, c, cov, &chisq, work);
	gsl_multifit_linear_free (work);

	printf("Answer %lf %lf %lf", gsl_vector_get(c,0), gsl_vector_get(c,1), gsl_vector_get(c,2));

	max_loc=1.0*gsl_vector_get(c,1)/(2.0*gsl_vector_get(c,2));
	printf("Maximized at %lf\n", max_loc);
	gsl_matrix_free (X);
	gsl_vector_free (y);
	gsl_vector_free (w);
	gsl_vector_free (c);
	gsl_matrix_free (cov);
	
	return max_loc;
}