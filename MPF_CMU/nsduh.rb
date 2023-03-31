#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o HIDDEN_FITS_SCAN_p2_NSDUH_LARGE -t 48:00:00 -p RM ./nsduh.rb
require 'parallel'

prefix=""
new_prefix=""

ans=[2.0, 1.0].collect { |p_norm|
  
  [0,1,2].collect { |hidden|

    filename="NSDUH_full.txt"

    file=File.new(prefix+filename, 'r')
    n_lines=file.readline.to_i
    n=file.readline.to_i

    print "Doing #{filename} at #{Time.now} with #{p_norm} and #{hidden} hidden nodes\n"
    filename_out=filename+".mpf"+"_p#{p_norm}_hidden#{hidden}"
    
    file=File.new(prefix+filename, 'r')
    file.readline
    file.readline
    str=""
    file.each_line { |set|
      str << ("X"*hidden)+set
    }
    file.close
    file_out=File.new(new_prefix+filename_out, 'w')
    file_out.write("#{n_lines}\n#{n+hidden}\n")
    file_out.write(str); file_out.close

    print "executing: OMP_NUM_THREADS=128 ./mpf -c #{new_prefix+filename_out} #{p_norm}\n"
    ans=`OMP_NUM_THREADS=128 ./mpf -c #{new_prefix+filename_out} #{p_norm}`
    print "FINAL VERSION for #{filename_out}:\n #{ans}\n\n"

    sparsity=ans.scan(/Best\ log\_sparsity\:[^\n]+\n/)[0].split(" ")[-1].to_f
    logl=ans.scan(Regexp.new(Regexp.escape(sparsity.to_s)+"[^\\n]+\\n"))[0].split(" ")[-1].to_f
    overall=ans.scan(/parameters\:[^\n]+\n/)[0].split(" ")[-1].to_f
    print "FINISHED AT #{Time.now}\n"

    ans=[p_norm, hidden, sparsity, logl, overall]
    print "#{ans}\n"

    final=Parallel.map(Array.new(128) { |i| i },  :in_process=>128) { |proc|
      ans=`./mpf -l #{new_prefix+filename_out} #{sparsity} #{p_norm}`
    }

    outcome=[p_norm, hidden] + final.collect { |ans| 
      [eval(ans.split("\n").select { |i| i.include?("params") }[0]), ans.scan(/parameters\:[^\n]+\n/)[0].split(" ")[-1].to_f]
    }.sort { |i,j| j[1] <=> i[1] }
    
    print "#{outcome}\n"
    outcome
  }
  
}
