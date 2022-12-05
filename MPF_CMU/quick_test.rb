#!/opt/local/bin/ruby
#!/usr/bin/ruby
# sbatch -N 1 -o quick_test_COMP -t 06:00:00 -p RM ./quick_test.rb

list=[]
ans=Array.new(100) { |i|

  `./mpf -g TEST_COMP/test_#{i} 20 256 0.5`

  `./mpf -z TEST_COMP/test_#{i}_params.dat 20`

  # `module load gcc`
  start=Time.now
  `./mpf -c TEST_COMP/test_#{i}_data.dat 1`
  gcc=Time.now-start
  `cp TEST_COMP/test_#{i}_data.dat_params.dat TEST_COMP/test_#{i}_data.dat_params_CV_GCC.dat`

  # `module load aocc aocl`
  start=Time.now
  `./mpf_AMD -c TEST_COMP/test_#{i}_data.dat 1`
  amd=Time.now-start
  
  `./mpf -z TEST_COMP/test_#{i}_data.dat_params.dat 20`

  `cp TEST_COMP/test_#{i}_data.dat_params.dat_probs.dat TEST_COMP/test_#{i}_data.dat_params.dat_probs_CV.dat`
  `cp TEST_COMP/test_#{i}_data.dat_params.dat TEST_COMP/test_#{i}_data.dat_params_CV.dat`

  `./mpf -l TEST_COMP/test_#{i}_data.dat -100 1`

  `./mpf -z TEST_COMP/test_#{i}_data.dat_params.dat 20`

  ans_gcc=`./mpf -k TEST_COMP/test_#{i}_data.dat TEST_COMP/test_#{i}_params.dat TEST_COMP/test_#{i}_data.dat_params_CV_GCC.dat`.split("\n")[0].split(":")[-1].to_f
  ans_amd=`./mpf -k TEST_COMP/test_#{i}_data.dat TEST_COMP/test_#{i}_params.dat TEST_COMP/test_#{i}_data.dat_params_CV.dat`.split("\n")[0].split(":")[-1].to_f

  print "GCC time: #{gcc} (#{ans_gcc})\n"
  print "AMD time: #{amd} (#{ans_amd})\n"
  
  list << [gcc, amd, ans_gcc, ans_amd]
  print "#{list}\n"
}
 
print "#{ans}\n"