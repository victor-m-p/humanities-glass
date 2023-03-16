#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o UPDATED_FITS_noNA -t 48:00:00 -p RM ./final_process.rb

prefix="/jet/home/sdedeo/humanities-glass/data/clean/"
new_prefix="/jet/home/sdedeo/humanities-glass/data/mdl_experiments/"

#[1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
ans=Array.new(21) { |i|
  2+i/20.0
}.collect { |p_norm|
  `ls #{prefix}`.split("\n").select { |filename|
    filename_out=filename+".mpf_p=#{p_norm}"

    n_lines=`wc -l #{prefix+filename}`.split(" ")[0].to_i
    n=filename.split("_")[2].to_i
    num_na=filename.split("_")[4].to_i
    
    (n == 20) and (num_na == 5)
  }.collect { |filename|
    n_lines=`wc -l #{prefix+filename}`.split(" ")[0].to_i
    n=filename.split("_")[2].to_i
    num_na=filename.split("_")[4].to_i
    
    print "Doing #{filename} at #{Time.now}\n"
    filename_out=filename+".mpf"+"p=#{p_norm}"
    
    file=File.new(prefix+filename, 'r')
    str=""
    file.each_line { |set|
      str << set.split(" ")[0..-2].collect { |i| (i.to_i == 0) ? "X" : ((i.to_i < 0) ? "0" : "1") }.join()+" "+set.split(" ")[-1]+"\n"
    }
    file.close
    file_out=File.new(new_prefix+filename_out, 'w')
    file_out.write("#{n_lines}\n#{n}\n")
    file_out.write(str); file_out.close

    ans=`OMP_NUM_THREADS=128 ./mpf -c #{new_prefix+filename_out} #{p_norm}`
    print "FINAL VERSION for #{filename_out}:\n #{ans}\n\n"
    ans.split("\n").select { |i| i.include?("params") }
    file_out=File.new(new_prefix+filename_out+"_params_NN1", 'w')
    file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
    file_out.close
    
    sparsity=ans.scan(/Best\ log\_sparsity\:[^\n]+\n/)[0].split(" ")[-1].to_f
    logl=ans.scan(Regexp.new(Regexp.escape(sparsity.to_s)+"[^\\n]+\\n"))[0].split(" ")[-1].to_f
    print "FINISHED AT #{Time.now}\n"
    [p_norm, sparsity, logl]
  }
}

print "#{ans}\n"