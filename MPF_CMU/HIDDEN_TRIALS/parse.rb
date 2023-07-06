load '../../../ENT/ent.rb'

base="region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt"
0.upto(7) { |n|
  file_in=File.new("../RAW/#{base}", 'r')
  file=File.new("../WORK_MULTI/#{base}_#{n}N", 'w')
  ntot=file_in.readline.to_i
  file.write("#{ntot}\n#{file_in.readline.to_i+n}\n")
  file_in.each_line { |line|
    file.write("#{line.split(" ")[0]+"X"*n} #{line.split(" ")[1]}\n")
  }
  file.close
  print "../mpf -l ../WORK_MULTI/#{base}_#{n}N 0.5 1.0 &\n"
}

## idea — keep iterating on number of hidden nodes until you start beating the KL for the base condition.

path="../WORK_MULTI"
set=Hash.new
Dir.glob(path+"/*params.dat").sort.each { |filename|
  file=File.new(filename, 'r'); ans=file.read.split(" ").collect { |i| i.to_f }
  n=38+filename.scan(/[0-9]N/)[0].gsub(/[^0-9]/, "").to_i
  count=0
  lookup=Hash.new
  0.upto(n-2) { |i|
    (i+1).upto(n-1) { |j|
      lookup[[i,j]]=ans[count]
      count += 1
    }
  }
  n.times { |i|
    lookup[i]=ans[count+i]
  }
  set[n-38]=lookup
}

set.keys.sort.collect { |i| [i, set[i].keys.select { |j| j.class == Array }.collect { |j| set[i][j].abs }.mean] }

0.upto(set.keys.max-1) { |n|
  (n+1).upto(set.keys.max) { |np|
    zero=set[n].keys.select { |j| j.class == Array and j[0] < 38 and j[1] < 38 }.collect { |i| set[n][i] }
    one=set[np].keys.select { |j| j.class == Array and j[0] < 38 and j[1] < 38 }.collect { |i| set[np][i] }
    print "#{n} #{np}: #{zero.correlation(one)}\n"    
  }
  print "\n"
}

Parallel.map(Dir.glob(path+"/*params.dat").sort.select { |i| !i.include?("samples") }, :in_process=>10) { |filename|
  n=38+filename.scan(/[0-9]N/)[0].gsub(/[^0-9]/, "").to_i
  `../mpf -s #{filename} #{n} 10000000 &`
}

sub_dist=Hash.new
Dir.glob(path+"/*dat").sort.select { |i| i.include?("samples") }.each { |filename|
  n=filename.scan(/[0-9]N/)[0].gsub(/[^0-9]/, "").to_i
  sub_dist[n]=Hash.new(0)
  file=File.new(filename, 'r')
  norm=1.0/`wc -l #{filename}`.split(" ")[0].to_f
  file.each_line { |line| sub_dist[n][line[0..37]] += norm }
};1

# 0.upto(set.keys.max-1) { |n|
#   print "#{n}: #{sub_dist[n].to_a.collect { |i| i[1] }.ent}\n"
#   (n).upto(set.keys.max) { |np|
#     tot=(sub_dist[n].keys | sub_dist[np].keys)
#     print "#{np}: #{tot.collect { |i| sub_dist[n][i]+1e-12 }.kl(tot.collect { |i| sub_dist[np][i]+1e-12 })}\n"
#   }
#   print "\n"
# }

data=Hash.new
file_in=File.new("../RAW/#{base}", 'r')
ntot=file_in.readline.to_f; file_in.readline
file_in.each_line { |line|
  data[line.split(" ")[0][0..37]] = line.split(" ")[1].to_f
};
0.upto(set.keys.max) { |n|
  kl=Parallel.map(data.keys, :in_process=>10) { |hit|
    rex=Regexp.new("^"+hit.gsub("X", ".")+"$")
    num=sub_dist[n].keys.select { |i| i.scan(rex).length > 0 };1
    ans=Math::log2(num.collect { |i| sub_dist[n][i] }.sum+1e-12)
    if num.length > 0 then
      print "#{hit} (#{data[hit]}) -- #{num.length}: #{ans}\n"
    end
    data[hit]*ans
  }.sum
  print "#{n}: #{kl}\n"
}

data.keys.collect { |i| i[0] }.select { |i| i != "X" }.collect { |i| i.to_f }.mean
0.upto(7) { |n|
  print "#{n}: #{sub_dist[n].keys.collect { |i| i[0].to_f*sub_dist[n][i] }.sum}\n"
}


# ./mpf -s WORK_MULTI/LAMBDA_-1/region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt_expanded_pnorm1_extra_0N_params.dat 38 10
