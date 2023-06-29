#!/usr/bin/ruby
#!/opt/local/bin/ruby
require 'parallel'

# sbatch -N 1 -o JUNE_FITS_0 -t 24:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_FITS_1 -t 24:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_FITS_2 -t 24:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_FITS_3 -t 24:00:00 -p RM ./simple_fits.rb 3

# sbatch -N 1 -o JUNE_FITS_N0 -t 24:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_FITS_N1 -t 24:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_FITS_N2 -t 24:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_FITS_N3 -t 24:00:00 -p RM ./simple_fits.rb 3

# sbatch -N 1 -o JUNE_FITS_NN0 -t 48:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_FITS_NN1 -t 48:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_FITS_NN2 -t 48:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_FITS_NN3 -t 48:00:00 -p RM ./simple_fits.rb 3

n=ARGV[0].to_i

filename=`ls RAW_2/*`.split("\n")[n]

transform="WORK_3/"+filename.split("/")[-1]

# Parallel.map(Array.new(128) { |i| i }, :in_process=>128) { |i|
#   lam=-1+2*(i+1)/128.0
#   `cp #{filename} #{transform+"_lam#{lam}_PNORM1"}`
#   ans=`./mpf -l #{transform+"_lam#{lam}_PNORM1"} #{lam} 1.0`
#   `rm #{transform+"_lam#{lam}_PNORM1"}`
#   print "lam=#{lam}\n#{ans}\n"
# }

`cp #{filename} #{transform+"_lam_CV_PNORM1"}`
ans=`./mpf -c #{transform+"_lam_CV_PNORM1"} 1.0`
`rm #{transform+"_lam_CV_PNORM2"}`
print "CV\n#{ans}\n"

# Parallel.map(Array.new(128) { |i| i }, :in_process=>128) { |i|
#   lam=-1+2*(i+1)/128.0
#   `cp #{filename} #{transform+"_lam#{lam}_PNORM2"}`
#   ans=`./mpf -l #{transform+"_lam#{lam}_PNORM2"} #{lam} 2.0`
#   `rm #{transform+"_lam#{lam}_PNORM2"}`
#   print "lam=#{lam}\n#{ans}\n"
# }

`cp #{filename} #{transform+"_lam_CV_PNORM2"}`
`./mpf -c #{transform+"_lam_CV_PNORM2"} 2.0`
`rm #{transform+"_lam_CV_PNORM2"}`
print "CV\n#{ans}\n"

