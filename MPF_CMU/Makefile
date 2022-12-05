SHELL=/bin/bash

# for mac...
CC=/opt/local/bin/clang

# for PSC with GCC
CC=gcc

# for PSC with AMD compiler
# run `module load aocc` first!
CC=clang

#fopenmp
CFLAGS=-O3 -mp -zopt -flto -fremap-arrays `gsl-config --cflags`
LFLAGS=`gsl-config --libs`

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
	rm -f mpf

	
