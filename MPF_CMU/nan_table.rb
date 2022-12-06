load '../../ENT/ent.rb'

list=[]
Dir.glob("*NAN_scan_20_5_*").collect { |f|
  file=File.open(f, 'r'); str=file.read; file.close
  ans=eval(str.scan(/^\[\[[^\n]+\n/)[-1])
}.each { |i|
  list += i
}

range=[[0.01,0.125], [0.125, 0.25], [0.25, 0.5], [0.5, 1.0]]
range.each { |i|
  set=list.select { |j| j[0] > i[0] and j[0] < i[1] }.transpose
  print "#{i} (#{set.transpose.length}) #{Array.new(4) { |k| set[k+1].mean } }\n"
};1
  
