#!/usr/bin/ruby

n=ARGV[0].to_i
label=ARGV[1]
`./mpf -g DATA/test_sequence_#{label} #{n} 1024 0.2`

file=File.new("DATA/test_sequence_#{label}_data.dat", 'r')
str=file.read; file.close

file=File.new("DATA/test_sequence_#{label}_base_data.dat", 'w')
str2="128\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

file=File.new("DATA/test_sequence_#{label}_256_data.dat", 'w')
str2="256\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

`./mpf -c DATA/test_sequence_#{label}_base_data.dat 1`
`./mpf -c DATA/test_sequence_#{label}_256_data.dat 1`
start=`./mpf -k DATA/test_sequence_#{label}base_data.dat DATA/test_sequence_#{label}params.dat DATA/test_sequence_#{label}base_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
best=`./mpf -k DATA/test_sequence_#{label}base_data.dat DATA/test_sequence_#{label}params.dat DATA/test_sequence_#{label}256_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

cut=1024
str_na=str.split("\n")[1..(128+1)].join("\n")+"\n"+str.split("\n")[129..(128+cut+1)].collect { |j| 
  loc=[]
  while(loc.length < 3) do
    while(loc.include?(pos=rand(n))) do     
    end
    loc << pos    
  end
  code=j.dup
  loc.each { |i|
    code[i]="X"
  }
  code
}.join("\n");1

[64, 128, 256, 512, 512+256, 1024].each { |cut|
  file=File.new("DATA/test_sequence_#{label}128_#{cut}NA3_data.dat", 'w')
  file.write("#{128+cut}\n"+str_na); file.close
  `./mpf -c DATA/test_sequence_#{label}128_#{cut}NA3_data.dat 1`  
  ans=`./mpf -k DATA/test_sequence_#{label}base_data.dat DATA/test_sequence_#{label}params.dat DATA/test_sequence_#{label}128_#{cut}NA3_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
  print "#{cut}: #{ans} (vs #{best}, vs #{start})\n"
}

# sbatch -N 1 -o NAN_TESTS_10_1 -t 2:00:00 -p RM ./test_nans.rb 10 1
# sbatch -N 1 -o NAN_TESTS_10_2 -t 2:00:00 -p RM ./test_nans.rb 10 2
# sbatch -N 1 -o NAN_TESTS_10_3 -t 2:00:00 -p RM ./test_nans.rb 10 3
# sbatch -N 1 -o NAN_TESTS_10_4 -t 2:00:00 -p RM ./test_nans.rb 10 4
# sbatch -N 1 -o NAN_TESTS_10_5 -t 2:00:00 -p RM ./test_nans.rb 10 5
# sbatch -N 1 -o NAN_TESTS_20_1 -t 2:00:00 -p RM ./test_nans.rb 20 6
# sbatch -N 1 -o NAN_TESTS_20_2 -t 2:00:00 -p RM ./test_nans.rb 20 7
# sbatch -N 1 -o NAN_TESTS_20_3 -t 2:00:00 -p RM ./test_nans.rb 20 8
# sbatch -N 1 -o NAN_TESTS_20_4 -t 2:00:00 -p RM ./test_nans.rb 20 9
# sbatch -N 1 -o NAN_TESTS_20_5 -t 2:00:00 -p RM ./test_nans.rb 20 10