#!/usr/bin/ruby
# sbatch -N 1 -o quick_test_COMP -t 06:00:00 -p RM ./quick_test.rb

ans=Array.new(100) { |i|

  `./mpf -g TEST_COMP/test_#{i} 20 256 0.5`

  `./mpf -z TEST_COMP/test_#{i}_params.dat 20`

  start=Time.now
  print "Doing GCC case; #{start}\n"
  `./mpf -c TEST_COMP/test_#{i}_data.dat 1`
  gcc=Time.now-start
  print "GCC case finish at #{Time.now-start}\n"
  `cp TEST_COMP/test_#{i}_data.dat_params.dat TEST_COMP/test_#{i}_data.dat_params_CV_GCC.dat`

  start=Time.now
  print "Doing AMD case; #{start}\n"
  `./mpf_AMD -c TEST_COMP/test_#{i}_data.dat 1`
  amd=Time.now-start
  print "AMD case finish at #{Time.now-start}\n"

  `./mpf -z TEST_COMP/test_#{i}_data.dat_params.dat 20`

  `cp TEST_COMP/test_#{i}_data.dat_params.dat_probs.dat TEST_COMP/test_#{i}_data.dat_params.dat_probs_CV.dat`
  `cp TEST_COMP/test_#{i}_data.dat_params.dat TEST_COMP/test_#{i}_data.dat_params_CV.dat`

  `./mpf -l TEST_COMP/test_#{i}_data.dat -100 1`

  `./mpf -z TEST_COMP/test_#{i}_data.dat_params.dat 20`

  `./mpf -k TEST_COMP/test_#{i}_data.dat TEST_COMP/test_#{i}_params.dat TEST_COMP/test_#{i}_data.dat_params_CV.dat`
  `./mpf -k TEST_COMP/test_#{i}_data.dat TEST_COMP/test_#{i}_params.dat TEST_COMP/test_#{i}_data.dat_params.dat`
  
  [gcc, amd]
}
 
print "#{ans}\n"