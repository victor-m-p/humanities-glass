#!/usr/bin/ruby

n=10
`./mpf -g DATA/test_sequence #{n} 1024 0.2`

file=File.new("DATA/test_sequence_data.dat", 'r')
str=file.read; file.close

file=File.new("DATA/test_sequence_base_data.dat", 'w')
str2="128\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

file=File.new("DATA/test_sequence_256_data.dat", 'w')
str2="256\n"+str.split("\n")[1..-1].join("\n");1
file.write(str2); file.close

`./mpf -c DATA/test_sequence_base_data.dat 1`
`./mpf -c DATA/test_sequence_256_data.dat 1`
start=`./mpf -k DATA/test_sequence_base_data.dat DATA/test_sequence_params.dat DATA/test_sequence_base_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
best=`./mpf -k DATA/test_sequence_base_data.dat DATA/test_sequence_params.dat DATA/test_sequence_256_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f

[128, 256, 512].each { |cut|
  file=File.new("DATA/test_sequence_128_#{cut}NA3_data.dat", 'w')
  str2="256\n"+str.split("\n")[1..(128+1)].join("\n")+"\n"+str.split("\n")[129..(128+cut+1)].collect { |j| 
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
  file.write(str2); file.close
  `./mpf -c DATA/test_sequence_128_#{cut}NA3_data.dat 1`  
  ans=`./mpf -k DATA/test_sequence_base_data.dat DATA/test_sequence_params.dat DATA/test_sequence_128_#{cut}NA3_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
  print "#{cut}: #{ans} (vs #{best})\n"
}



