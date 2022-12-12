#!/user/bin/env bash
for nq in 20 30 40
do
	for nn in 0 1 2 3 4 5 6 7 8 9 10
	do
		python DRH_curate.py \
		-i ../data/raw/drh_20221019.csv \
		-om ../data/clean/ \
		-or ../data/reference/ \
		-nq $nq \
		-nn $nn
	done
done
