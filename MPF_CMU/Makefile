SHELL=/bin/bash

# for mac...
CC=/opt/local/bin/clang

# for PSC with GCC
# run `module load gcc` first!
CC=gcc
CFLAGS=-O3 -fopenmp `gsl-config --cflags`

# for PSC with AMD compiler
# run `module load aocc aocl` first!
# CC=clang
# CFLAGS=-O3 -fopenmp -fvectorize -flto -fremap-arrays -funroll-loops -finline-aggressive -lalm  -lm `gsl-config --cflags`

#fopenmp
LFLAGS=`gsl-config --libs`

OBJ = mpf.o mcmc.o nlin.o nsb.o

all : main.c $(OBJ)
	${CC} $^ -o mpf ${CFLAGS} ${LFLAGS}

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

	
