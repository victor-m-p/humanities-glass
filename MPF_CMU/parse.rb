#!/usr/bin/ruby

filename="test_new_data.dat"

file=File.new(filename, 'r')
m=file.readline.to_i
n=file.readline.to_i

file_out=File.new(filename+"_HIDDEN", 'w')
file_out_removed=File.new(filename+"_REMOVED", 'w')
file_out.write("#{m}\n#{n}\n")
file_out_removed.write("#{m}\n#{n-1}\n")
file.each_line { |line|
  file_out.write("X#{line[1..-1]}")
  file_out_removed.write("#{line[1..-1]}")
}
file_out.close
file_out_removed.close
