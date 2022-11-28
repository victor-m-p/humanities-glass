#!/user/bin/env bash
for nq in 20 30
do
	for nn in 0 1 2 3 4 5 6 7 8 9 10
	do
		python lexibank_curate.py \
		-i ../data/lexibank/raw/lexicon-values.csv \
		-om ../data/lexibank/clean/ \
		-or ../data/lexibank/reference/ \
		-nq $nq \
		-nn $nn
	done
done
