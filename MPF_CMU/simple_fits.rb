#!/usr/bin/ruby
#!/opt/local/bin/ruby
require 'parallel'

# sbatch -N 1 -o JUNE_NEW_0 -t 48:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_NEW_1 -t 48:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_NEW_2 -t 48:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_NEW_3 -t 48:00:00 -p RM ./simple_fits.rb 3
# sbatch -N 1 -o JUNE_NEW_4 -t 48:00:00 -p RM ./simple_fits.rb 4
# sbatch -N 1 -o JUNE_NEW_5 -t 48:00:00 -p RM ./simple_fits.rb 5
# sbatch -N 1 -o JUNE_NEW_6 -t 48:00:00 -p RM ./simple_fits.rb 6
# sbatch -N 1 -o JUNE_NEW_7 -t 48:00:00 -p RM ./simple_fits.rb 7

n=ARGV[0].to_i

filename=`ls RAW/*`.split("\n")[n]

transform="WORK/"+filename.split("/")[-1]

print "#{filename}\n"

`cp #{filename} #{transform+"_lam_CV_PNORM1"}`
ans=`./mpf -c #{transform+"_lam_CV_PNORM1"} 1.0`
`rm #{transform+"_lam_CV_PNORM2"}`
print "CV\n#{ans}\n"