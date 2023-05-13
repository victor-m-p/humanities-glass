load '../../ENT/ent.rb'
require 'parallel'

`./mpf -g test 10 1000 0.25`
file=File.new("test_data.dat", 'r')
str=file.read; file.close

Array.new(31) { |sp_i|
  sp=sp_i/10.0-2.0
  
  tot=Parallel.map(Array.new(10) { |i| i },  :in_process=>10) { |process|
    
    file=File.new("test_data_HIDDEN_#{process}.dat", 'w')
    ans=str.split("\n")
    file.write("500\n#{ans[1]}\n")
    ans[2..-501].each { |i|
      file.write("#{i.split(" ")[0][0..-4]}XXX 1.0\n")
    }; file.close
    
    ans_out=`./mpf -l test_data_HIDDEN_#{process}.dat #{sp} #{1}`
    logl=ans_out.split("\n")[-2].split(" ")[-1].to_f
    params=eval(ans_out.split("\n")[5])

    # file=File.new("test_data_HIDDEN_#{process}_NEW.dat", 'w')
    # ans=str.split("\n")
    # file.write("500\n#{ans[1]}\n")
    # ans[2+500..-1].each { |i|
    #   file.write("#{i.split(" ")[0][0..-4]}XXX 1.0\n")
    # }; file.close

    ans_out=`./mpf -k test_data_HIDDEN_#{process}.dat test.dat_params.dat test_data_HIDDEN_#{process}.dat_params.dat`
    logl=ans_out.split("\n")[0].split(" ")[-1].to_f
    
    logl
  }.sort { |i,j| i <=> j }
  final=[sp, tot[0], tot.collect { |j| j }.mean]
  print "#{final}\n"
  final
}
