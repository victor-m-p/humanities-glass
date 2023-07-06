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

lookup=set[2];
file=File.new("2N.csv", 'w')
file.write("Source,Target,Weight,Type\n")
lookup.keys.select { |i| i.class == Array }.each { |i|
  file.write("#{i[0]},#{i[1]},#{lookup[i].abs},#{lookup[i] < 0 ? -1 : 1}\n")
}
file.close

file=File.new("nodes.csv", 'w')
file.write("Id,Label,Type\n")
(38+2).times { |i|
  file.write("#{i},#{i},#{i < 38 ? 1 : 0}\n")
}
file.close

lookup=set[0];
file=File.new("0N.csv", 'w')
file.write("Source,Target,Weight,Type\n")
lookup.keys.select { |i| i.class == Array }.each { |i|
  file.write("#{i[0]},#{i[1]},#{lookup[i].abs},#{lookup[i] < 0 ? -1 : 1}\n")
}
38.upto(38+2-1) { |i|
  0.upto(38+2-1) { |j|
    file.write("#{i},#{j},#{0},#{-1}\n")    
  }
}
file.close
