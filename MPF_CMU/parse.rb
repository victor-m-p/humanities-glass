#!/usr/bin/ruby

# sbatch -N 1 -o EXPERIMENTS_2X -t 48:00:00 -p RM ./parse.rb

filename="../data/mdl_experiments/matrix_questions_20_maxna_5_nrows_455_entries_407.txt.mpf"

file=File.new(filename, 'r')
m=file.readline.to_i
n=file.readline.to_i

file_out=File.new(filename+"_HIDDEN", 'w')
file_out_removed=File.new(filename+"_REMOVED", 'w')

file_out_added=File.new(filename+"_ADDED", 'w')
file_out_added_2=File.new(filename+"_ADDED_2X", 'w')
file_out_added_3=File.new(filename+"_ADDED_3X", 'w')
file_out_added_4=File.new(filename+"_ADDED_4X", 'w')

file_out.write("#{m}\n#{n}\n")
file_out_removed.write("#{m}\n#{n-1}\n")
file_out_added.write("#{m}\n#{n+1}\n")
file_out_added_2.write("#{m}\n#{n+2}\n")
file_out_added_3.write("#{m}\n#{n+3}\n")
file_out_added_4.write("#{m}\n#{n+4}\n")
file.each_line { |line|
  file_out.write("X#{line[1..-1]}")
  file_out_removed.write("#{line[1..-1]}")
  file_out_added.write("X#{line[0..-1]}")
  file_out_added_2.write("XX#{line[0..-1]}")
  file_out_added_3.write("XXX#{line[0..-1]}")
  file_out_added_4.write("XXXX#{line[0..-1]}")
}
file_out.close
file_out_removed.close
file_out_added.close
file_out_added_2.close
file_out_added_3.close
file_out_added_4.close

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_ADDED_2X"}`
print "ADDED 2 FINISHED\n"
print "#{val}\n"

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_ADDED_3X"}`
print "ADDED 3 FINISHED\n"
print "#{val}\n"

val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_ADDED_4X"}`
print "ADDED 4 FINISHED\n"
print "#{val}\n"

# val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_ADDED"}`
# print "ADDED FINISHED\n"
# print "#{val}\n"
#
# val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_HIDDEN"}`
# print "HIDDEN FINISHED\n"
# print "#{val}\n"
#
# val=`OMP_NUM_THREADS=128 ./mpf -c #{filename+"_REMOVED"}`
# print "REMOVED FINISHED\n"
# print "#{val}\n"


