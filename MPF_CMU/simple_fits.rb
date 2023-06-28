#!/usr/bin/ruby
#!/opt/local/bin/ruby
require 'parallel'

# sbatch -N 1 -o JUNE_FITS_0 -t 48:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_FITS_1 -t 48:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_FITS_2 -t 48:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_FITS_3 -t 48:00:00 -p RM ./simple_fits.rb 3

n=ARGV[0].to_i

filename=`ls ../data/religious-landscapes/*`.split("\n")[n]

Parallel.map(Array.new(128) { |i| i }, :in_process=>128) { |i|
  lam=-1+2*(i+1)/128.0
  `cp #{filename} #{filename+"_lam#{lam}_PNORM1"}`
  ans=`./mpf -l #{filename+"_lam#{lam}_PNORM1"} #{lam} 1.0`
  `rm #{filename+"_lam#{lam}_PNORM1"}`
  print "lam=#{lam}\n#{ans}\n"
}

`cp #{filename} #{filename+"_lam_CV_PNORM1"}`
ans=`./mpf -c #{filename+"_lam_CV_PNORM1"} 1.0`
`rm #{filename+"_lam_CV_PNORM1"}`
print "CV\n#{ans}\n"

Parallel.map(Array.new(128) { |i| i }, :in_process=>128) { |i|
  lam=-1+2*(i+1)/128.0
  `cp #{filename} #{filename+"_lam#{lam}_PNORM2"}`
  ans=`./mpf -l #{filename+"_lam#{lam}_PNORM2"} #{lam} 2.0`
  `rm #{filename+"_lam#{lam}_PNORM2"}`
  print "lam=#{lam}\n#{ans}\n"
}

`cp #{filename} #{filename+"_lam_CV_PNORM2"}`
`./mpf -c #{filename+"_lam_CV_PNORM2"} 2.0`
`rm #{filename+"_lam_CV_PNORM1"}`
print "CV\n#{ans}\n"

