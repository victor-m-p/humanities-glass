["region_World_Region_questions_20_nan_5_rows_2350_entries_354.txt", "region_NGA_questions_20_nan_5_rows_3350_entries_354.txt", "region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt", "region_NGA_questions_38_nan_5_rows_5097_entries_214.txt"].each { |filename|
  file=File.new("RAW/#{filename}", 'r')
  file_new=File.new("RAW/#{filename}_expanded", 'w')
  n=file.readline.to_i
  node=file.readline.to_i
  file_new.write("#{n}\n#{node+5}\n")
  file.each_line { |line|
    set=line.split(" ")
    file_new.write("#{set[0]+"X"*5} #{set[1]}\n")
  }
  file_new.close
}


["region_World_Region_questions_20_nan_5_rows_2350_entries_354.txt", "region_NGA_questions_20_nan_5_rows_3350_entries_354.txt", "region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt", "region_NGA_questions_38_nan_5_rows_5097_entries_214.txt"].each { |filename|
  2.times { |run|
    `cp RAW/#{filename}_expanded WORK_MULTI/#{filename}_expanded_pnorm1_RUN#{run}`
  }
}

["region_World_Region_questions_20_nan_5_rows_2350_entries_354.txt", "region_NGA_questions_20_nan_5_rows_3350_entries_354.txt", "region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt", "region_NGA_questions_38_nan_5_rows_5097_entries_214.txt"].each { |filename|
  2.times { |run|
    print "./mpf -l WORK_MULTI/#{filename}_expanded_pnorm1_RUN#{run} -1.0 1.0 &\n"
  }
}

`./mpf -l WORK_MULTI/#{filename}_expanded_pnorm1_RUN#{run} -1.0 1.0 &` 
`rm WORK_MULTI/#{filename}_expanded_pnorm1_RUN#{run}`

file=File.new("WORK/region_World_Region_questions_38_nan_5_rows_3231_entries_300.txt_lam-1_PNORM1_params.dat", 'r'); ans=file.read.split(" ").collect { |i| i.to_f }
n=38
count=0
lookup=Hash.new
0.upto(n-2) { |i|
  (i+1).upto(n-1) { |j|
    lookup[[i,j]]=ans[count]
    count += 1
  }
}

file=File.new("BASIC_region_World_Region_questions_38_nan_5_rows_3231_entries_300_edges.csv", 'w')
file.write("Source,Target,Weight,Flag\n")
lookup.keys.each { |i|
  file.write("#{i[0]},#{i[1]},#{lookup[i].abs},#{lookup[i] < 0 ? -1 : 1}\n")
};file.close

filename="region_World_Region_questions_38_nan_5_rows_2692_entries_214.txt"
run=1
file=File.new("WORK_MULTI/#{filename}_expanded_pnorm1_RUN#{run}_params.dat", 'r'); ans=file.read.split(" ").collect { |i| i.to_f }
n=38+5
count=0
lookup_hidden=Hash.new
0.upto(n-2) { |i|
  (i+1).upto(n-1) { |j|
    lookup_hidden[[i,j]]=ans[count]
    count += 1
  }
}

file=File.new("#{filename}_#{run}RUN_edges.csv", 'w')
file.write("Source,Target,Weight,Flag\n")
lookup.keys.each { |i|
  file.write("#{i[0]},#{i[1]},#{lookup[i].abs},#{lookup[i] < 0 ? -1 : 1}\n")
};file.close

file=File.new("nodes.csv", 'w')
file.write("Id,Label,Type\n")
n.times { |i|
  file.write("#{i},#{i},#{i < 38 ? 0 : 1}\n")
}
file.close


