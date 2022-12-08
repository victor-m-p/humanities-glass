#!/usr/bin/ruby
# sbatch -N 7 -o NAN_TESTS/new_NAN_TESTS_final -t 24:00:00 -p RM ./test_nans.rb 20 5 NAN_TESTS_FINAL
# [sdedeo@bridges2-login013 MPF_CMU]$ sbatch -N 1 -o NAN_TESTS/new_NAN_TESTS_final -t 24:00:00 -p RM ./test_nans.rb 20 5 NAN_TESTS_FINAL
# Submitted batch job 13527344
# sbatch -N 1 -o NAN_TESTS/new_NAN_TESTS_final_1 -t 12:00:00 -p RM ./test_nans.rb 20 5 NAN_TESTS_FINAL_1

n=ARGV[0].to_i
nan=ARGV[1].to_i
label=ARGV[2]
n_proc=7
require 'parallel'

10.times { |i|
  print "Starting new test #{i} at time #{Time.now}...\n"
  
  `./mpf -g DATA/test_sequence_#{label} #{n} 2048 0.2`

  full_data=Parallel.map([0, 64, 128, 256, 512, 512+256, 1024], :in_process=>n_proc) { |cut|
    file=File.new("DATA/test_sequence_#{label}_data.dat", 'r')
    str=file.read; file.close

    file=File.new("DATA/test_sequence_#{label}_#{cut+128}_data.dat", 'w')
    str2="#{cut+128}\n"+str.split("\n")[1..-1].join("\n");1
    file.write(str2); file.close

    `./mpf -c DATA/test_sequence_#{label}_#{cut+128}_data.dat 1`
    start=`./mpf -k DATA/test_sequence_#{label}_#{cut+128}_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_#{cut+128}_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
    print "Full data -- #{cut}: #{start}\n"
    [cut, start]    
  }

  str_na=str.split("\n")[1..129].join("\n")+"\n"+str.split("\n")[130..-1].collect { |j| 
    loc=[]
    while(loc.length < nan) do
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

  nan_data=Parallel.map([0, 64, 128, 256, 512, 512+256, 1024], :in_process=>n_proc) { |cut| #, 512, 512+256, 1024
    file=File.new("DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat", 'w')
    file.write("#{128+cut}\n"+str_na); file.close
    `./mpf -c DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat 1`
    begin
      ans=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat_params.dat`
      ans=ans.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
      print "NA data -- #{cut}: #{ans}\n"
      [cut, ans]
    rescue
      print "Something bad happened at #{cut}\n"   
      [cut, -1] 
    end
  }

  print "Now do bad choice...\n"
  avg=Array.new(n) { 0 }
  str.split("\n")[2..(128+1)].each { |i|
    set=i.split(" ")[0].split("")
    n.times { |k|
      avg[k] += set[k].to_f
    }
  };
  avg.collect! { |i| i/128.0 }
  str_na_new=str_na.split("\n")[1..-1].collect { |j| 
    code=j.dup
    Array.new(n) { |i| code[i] == "X"  ? avg[i].round.to_s : code[i] }.join("")+" 1.0"
  }.join("\n");1

  bad_data=Parallel.map([0, 64, 128, 256, 512, 512+256, 1024], :in_process=>n_proc) { |cut| #, 512, 512+256, 1024
    file=File.new("DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat", 'w')
    file.write("#{128+cut}\n#{n}\n"+str_na_new); file.close
    `OMP_NUM_THREADS=128 ./mpf -c DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat 1` 
    begin
      ans=`./mpf -k DATA/test_sequence_#{label}_base_data.dat DATA/test_sequence_#{label}_params.dat DATA/test_sequence_#{label}_128_#{cut}NA#{nan}_data.dat_params.dat`.scan(/KL:[^\n]+\n/)[0].split(" ")[-1].to_f
      print "Bad data -- #{cut}: #{ans}\n"
      [cut, ans]
    rescue
      print "Something bad happened at #{cut}\n"
      [cut, -1]
    end
  }
  
  print "Finished test at #{Time.now}\n"
  print "full_data=#{full_data}\n"
  print "nan_data=#{nan_data}\n"
  print "bad_data=#{bad_data}\n"
}

# 3.times { |label|
#   [20].each { |nodes|
#     [5].each { |nan|
#       print "sbatch -N 1 -o DATA/new_NAN_TESTS_#{nodes}nodes_#{nan}NAN_#{label}_DDLONG -t #{nodes == 10 ? "02:00" : "24:00"}:00 -p RM ./test_nans.rb #{nodes} #{nan} #{label}_#{nodes}_#{nan}_DDLONG\n"
#     }
#   }
# }
