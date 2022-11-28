#!/usr/bin/ruby
#!/opt/local/bin/ruby
#sbatch -N 1 -o CV_DATA --mail-type=ALL -t 12:00:00 -p RM ./cv_data.rb

preface="../data/clean/"
preface_new="../data/mdl/"

set=`ls -alh /Users/simon/Desktop/humanities-glass/data/clean`.split("\n").collect { |i| i.split(" ")[-1] }.select { |i| i.include?("txt") and !i.include?("mpf") }.collect { |i| 
  n=i.split("_")[6].to_i
  na=i.split("_")[-1].to_i
  [n, na, i]
}.sort { |i,j| (i[0] <=> j[0]) == 0 ? i[1] <=> j[1] : i[0] <=> j[0] }

nn=1
set.each { |trial|

  n_lines=`wc -l #{preface+trial[-1]}`.split(" ")[0].to_i
  n=trial[0]
  file=File.new(preface+trial[-1], 'r')
  str=""
  file.each_line { |set|
    str << set.split(" ")[0..-2].collect { |i| (i.to_i == 0) ? "X" : ((i.to_i < 0) ? "0" : "1") }.join()+" "+set.split(" ")[-1]+"\n"
  }
  file.close

  file_out=File.new(preface_new+trial[-1]+".mpf", 'w')
  file_out.write("#{n_lines}\n#{n}\n")
  file_out.write(str); file_out.close

  ans=`./mpf -c #{preface_new+trial[-1]+".mpf"} #{nn}`
  print "#{ans}\n\n"
  best_log_sparsity=ans.scan(/Best\ log\_sparsity\:[^n]+\n/)[-1].split(" ")[-1].to_f
  file_out=File.new(preface_new+trial[-1]+".mpf_params_NN#{nn}_LAMBDA#{best_log_sparsity}", 'w')
  file_out.write(ans.split("\n").select { |i| i.include?("params") }[0])
  file_out.close

  print "n=#{trial[0]}; maxNA=#{trial[1]}; NN=#{nn}; Best Log-Sparsity: #{best_log_sparsity}; #{ans.split("\n").select { |i| i.include?("Clock") }[0]}\n"

  `cd /Users/simon/Desktop/humanities-glass ; git add . ; git commit -m "updating files #{Time.now}" ; git push`
}

