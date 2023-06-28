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

[20].each { |n|
  n_samp=500 #+2*10

  str="#{n_samp}\n#{n+1}\n"
  n_samp.times { |i|
    str << "X#{"#{i.modulo(2)}" * n} 1.0\n"
  }
  # str << ("1"*(n+1)+" 1.0\n")*10
  # str << ("0"*(n+1)+" 1.0\n")*10
  
  file=File.new("test.dat", 'w'); file.write(str); file.close

  lookup=Hash.new
  count=0
  (n).times { |i|
    (i+1).upto(n) { |j|
      lookup[[i,j]]=count
      count += 1
    }
  }

  print "DOING #{n}: \n"
  final=Array.new(11) { |spp|
    sp=4*(spp/10.0)-2
    ans=Parallel.map(Array.new(10) { |i| i },  :in_process=>10) { |i|
      params=eval(`./mpf -l test.dat #{sp} 1.0`.split("\n")[-3])
      hid=lookup.keys.select { |j| j.include?(0) }.collect { |p|
       params[lookup[p]].abs 
      }.mean
      vis=lookup.keys.select { |j| !j.include?(0) }.collect { |p|
       params[lookup[p]].abs
      }.mean
      [hid, vis, params[(n+1)*n/2..-1].collect { |j| j }.mean]
    }.transpose.collect { |i| i.mean }
    tot=ans+[sp]
    print "#{tot}\n"
    tot
  }
}





