#!/usr/bin/ruby
# sbatch -N 1 -o quick_test_OUT -t 0030:00 -p RM ./quick_test.rb

`./mpf -g TEST/test 20 256 0.5`

`./mpf -z TEST/test_params.dat 20`

start=Time.now
print "Doing GCC case; #{start}\n"
`./mpf -c TEST/test_data.dat 1`
print "Finish at #{Time.now-start}\n"
`cp TEST/test_data.dat_params.dat TEST/test_data.dat_params_CV_GCC.dat`

start=Time.now
print "Doing AMD case; #{start}\n"
`./mpf_AMD -c TEST/test_data.dat 1`
print "Finish at #{Time.now-start}\n"

`./mpf -z TEST/test_data.dat_params.dat 20`

`cp TEST/test_data.dat_params.dat_probs.dat TEST/test_data.dat_params.dat_probs_CV.dat`
`cp TEST/test_data.dat_params.dat TEST/test_data.dat_params_CV.dat`

`./mpf -l TEST/test_data.dat -100 1`

`./mpf -z TEST/test_data.dat_params.dat 20`

`./mpf -k TEST/test_data.dat TEST/test_params.dat TEST/test_data.dat_params_CV.dat`
`./mpf -k TEST/test_data.dat TEST/test_params.dat TEST/test_data.dat_params.dat`

