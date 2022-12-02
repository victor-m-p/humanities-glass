class String
  def hamming(b)
    h=0
    self.length.times { |i|
      if self.split("")[i] != b[i] then
        h += 1
      end
    }
    h
  end
end

file=File.new("cleaned_nrows_455_maxna_5.dat_params.dat_probs.dat", 'r')
set=Hash.new
file.each_line { |i|
  set[i.split(" ")[0]]=i.split(" ")[2].to_f
}; file.close

top20=set.to_a.sort { |i,j| j[1] <=> i[1] }[0..49]
top20_sav=top20.collect { |i| i.dup };1

final_list=[]
i=0
while(i < top20.length) do
  unit=[top20[i]]
  k=(i+1)
  while(k < top20.length) do
    if top20[i][0].hamming(top20[k][0]) < 2 then
      print "hit! #{i} and #{k}\n"
      unit << top20[k]
      top20.delete_at(k)
    else
      k += 1
    end
  end
  final_list << unit
  i += 1
end

cg_list=final_list[0..19].collect { |i|
  [i[0][0], i.collect { |j| j[1] }.sum]
}

max_h=[]
cg_list.length.times { |i|
  unit=[]
  0.upto(cg_list.length-1) { |j|
    unit << cg_list[i][0].hamming(cg_list[j][0])
  }
  max_h << (unit-[0]).min
}
max_dist=5

file=File.new("cg_edges.csv", 'w')
file.write("Source,Target,Weight\n")
(cg_list.length-1).times { |i|
    (i+1).upto(cg_list.length-1) { |j|
      if (max_dist-cg_list[i][0].hamming(cg_list[j][0])) > 0 then
        file.write("#{i},#{j},#{max_dist-cg_list[i][0].hamming(cg_list[j][0])}\n")
      end
    }
}
file.close

file=File.new("cg_nodes.csv", 'w')
file.write("Id,Label,Weight\n")
cg_list.length.times { |i|
  file.write("#{i},#{cg_list[i][0]},#{cg_list[i][1]}\n")
}
file.close









