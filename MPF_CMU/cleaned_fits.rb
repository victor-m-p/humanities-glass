#!/usr/bin/ruby
# sbatch -N 1 -o CLEANED_FITS -t 3:00:00 -p RM ./cleaned_fits.rb

`ls ../data/mdl_final/cleaned_*`.split("\n").each { |file|
  start=Time.now
  print "Doing #{file}...\n"
  `OMP_NUM_THREADS=128 ./mpf -c ../data/mdl_final/cleaned_maxna_#{i}.dat 1`  
  print "#{Time.now-start} seconds\n"

  `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "cleaned no-text fits (from PSC)" ; git push`
}
