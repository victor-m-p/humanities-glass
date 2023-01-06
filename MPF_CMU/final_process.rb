#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o UPDATED_FITS -t 24:00:00 -p RM ./final_process.rb

prefix="../data/clean/"
new_prefix="../data/mdl_experiments/"

`ls ../data/clean`.split("\n").each { |filename|
  filename_out=filename+".mpf"
  
  n_lines=`wc -l #{prefix+filename}`.split(" ")[0].to_i
  n=trial[0]
  file=File.new(filename, 'r')
  str=""
  file.each_line { |set|
    str << set.split(" ")[0..-2].collect { |i| (i.to_i == 0) ? "X" : ((i.to_i < 0) ? "0" : "1") }.join()+" "+set.split(" ")[-1]+"\n"
  }
  file.close
  file_out=File.new(new_prefix+filename_out, 'w')
  file_out.write("#{n_lines}\n#{n}\n")
  file_out.write(str); file_out.close

  ans=`./mpf -c #{new_prefix+filename_out} 1`
  ans.split("\n").select { |i| i.include?("params") }
  file_out=File.new(new_prefix+filename_out+"_params_NN1", 'w')
  file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
  file_out.close
  
  `cd /jet/home/sdedeo/humanities-glass ; git add . ; git commit -m "new cross-validated fits (from PSC)" ; git push`
}