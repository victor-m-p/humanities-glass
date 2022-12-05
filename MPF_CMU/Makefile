SHELL=/bin/bash

# for mac...
CC=/opt/local/bin/clang
CFLAGS=-O3 -fopenmp `gsl-config --cflags`
LFLAGS=-lm `gsl-config --libs`

# for PSC with GCC
# run `module load gcc` first!
# CC=gcc
# CFLAGS=-O3 -fopenmp `gsl-config --cflags`
# LFLAGS=`gsl-config --libs`

# for PSC with AMD compiler
# run `module load aocc aocl` first!
CC=clang
CFLAGS=-O3 -fopenmp -fvectorize -flto -fwhole-program-vtables -fremap-arrays -funroll-loops -fslp-vectorize -floop-unswitch-aggressive -finline-aggressive -mllvm --reduce-array-computations=3 `gsl-config --cflags`
LFLAGS=-unroll-aggressive -lalm -lamdlibm  -lm `gsl-config --libs`
cat
OBJ = mpf.o mcmc.o nlin.o nsb.o

all : main.c $(OBJ)
	${CC} $^ -o mpf_AMD ${CFLAGS} ${LFLAGS}

mpf.o: mpf.c
	${CC} $^ ${CFLAGS} -c

mcmc.o: mcmc.c
	${CC} $^ ${CFLAGS} -c

nlin.o: nlin.c
	${CC} $^ ${CFLAGS} -c

nsb.o: nsb.c
	${CC} $^ ${CFLAGS} -c

clean:
	rm -f *.o
	# rm -f mpf

	
