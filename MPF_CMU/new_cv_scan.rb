#!/usr/bin/ruby
# sbatch -N 1 -o NEW_CV_SCAN -t 12:00:00 -p RM ./new_cv_scan.rb

list=[]
start=Time.now
1000.times { |pos|
  beta=0
  while(beta < 0.001) do
    beta=((rand() < 0.5) ? rand()**2 : rand())
  end
  
  `./mpf -g test 20 128 #{beta}`
  ans=`./mpf -o test 1`
  val=[beta]+eval(ans.split("\n").select { |i| i.include?("val=") }[0].split("=")[-1])
  list << val
  print "#{(Time.now-start)/(pos+1)} per loop.\n"
  print "#{val}\n"
  file=File.new("cv_full.dat", 'w')
  file.write(Marshal.dump(list))
  file.close  
}
