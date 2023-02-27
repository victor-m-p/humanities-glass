#!/usr/bin/ruby

filename="../data/mdl_experiments/matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf"

file=File.new(filename, 'r')
m=file.readline.to_i
n=file.readline.to_i

file_out=File.new(filename+"_HIDDEN", 'w')
file_out_removed=File.new(filename+"_REMOVED", 'w')
file_out_added=File.new(filename+"_ADDED", 'w')

file_out.write("#{m}\n#{n}\n")
file_out_removed.write("#{m}\n#{n-1}\n")
file_out_added.write("#{m}\n#{n+1}\n")
file.each_line { |line|
  file_out.write("X#{line[1..-1]}")
  file_out_removed.write("#{line[1..-1]}")
  file_out_added.write("X#{line[0..-1]}")
}
file_out.close
file_out_removed.close

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_ADDED"}`
print "ADDED FINISHED\n"
print "#{val}\n"

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_HIDDEN"}`
print "HIDDEN FINISHED\n"
print "#{val}\n"

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_REMOVED"}`
print "REMOVED FINISHED\n"
print "#{val}\n"

