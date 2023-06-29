require 'parallel'

list=Parallel.map(`ls WORK/region_World_Region_questions_20_nan_5*`.split("\n"), :in_process=>10) { |i|
  base_file="RAW/"+i.split("/")[-1].split(".txt")[0]+".txt"
  ans=`./mpf -p #{base_file} 20 #{i}`  
  logl=ans.split("\n")[0].split(": ")[-1].to_f
  tot=[i, logl]
  print "#{tot}\n"
  tot
}
