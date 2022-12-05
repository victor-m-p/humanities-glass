#!/usr/bin/ruby
# sbatch -N 1 -o quick_test_#{i}_OUT -t 0030:00 -p RM ./quick_test.rb

ans=Parallel.map(Array.new(100) { |i| i }, in_processes: 100) { |i|

  `./mpf -g TEST/test_#{i} 20 256 0.5`

  `./mpf -z TEST/test_#{i}_params.dat 20`

  start=Time.now
  print "Doing GCC case; #{start}\n"
  `./mpf -c TEST/test_#{i}_data.dat 1`
  gcc=Time.now-start
  print "Finish at #{Time.now-start}\n"
  `cp TEST/test_#{i}_data.dat_params.dat TEST/test_#{i}_data.dat_params_CV_GCC.dat`

  start=Time.now
  print "Doing AMD case; #{start}\n"
  `./mpf_AMD -c TEST/test_#{i}_data.dat 1`
  amd=Time.now-start
  print "Finish at #{Time.now-start}\n"

  `./mpf -z TEST/test_#{i}_data.dat_params.dat 20`

  `cp TEST/test_#{i}_data.dat_params.dat_probs.dat TEST/test_#{i}_data.dat_params.dat_probs_CV.dat`
  `cp TEST/test_#{i}_data.dat_params.dat TEST/test_#{i}_data.dat_params_CV.dat`

  `./mpf -l TEST/test_#{i}_data.dat -100 1`

  `./mpf -z TEST/test_#{i}_data.dat_params.dat 20`

  `./mpf -k TEST/test_#{i}_data.dat TEST/test_#{i}_params.dat TEST/test_#{i}_data.dat_params_CV.dat`
  `./mpf -k TEST/test_#{i}_data.dat TEST/test_#{i}_params.dat TEST/test_#{i}_data.dat_params.dat`
  
  [gcc, amd]
}
 
print "#{ans}\n"