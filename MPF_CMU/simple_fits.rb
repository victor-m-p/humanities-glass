#!/usr/bin/ruby
#!/opt/local/bin/ruby

# sbatch -N 1 -o JUNE_FITS_0 -t 48:00:00 -p RM ./simple_fits.rb 0
# sbatch -N 1 -o JUNE_FITS_1 -t 48:00:00 -p RM ./simple_fits.rb 1
# sbatch -N 1 -o JUNE_FITS_2 -t 48:00:00 -p RM ./simple_fits.rb 2
# sbatch -N 1 -o JUNE_FITS_3 -t 48:00:00 -p RM ./simple_fits.rb 3

n=ARGV[0].to_i

filename=`ls ../data/religious-landscapes/*`.split("\n")[n]

Array.new(128) { |i|
  lambda=-1+2*(i+1)/128.0
  ans=`mpf -l #{filename+"_LAMBDA#{lambda}_PNORM1"} #{lambda} 1.0`
  print "LAMBDA=#{lambda}\n#{ans}\n"
}

ans=`mpf -c #{filename+"_LAMBDA_CV_PNORM1"} 1.0`
print "CV\n#{ans}\n"

Array.new(128) { |i|
  lambda=-1+2*(i+1)/128.0
  `mpf -l #{filename+"_LAMBDA#{lambda}_PNORM2"} #{lambda} 2.0`
  print "LAMBDA=#{lambda}\n#{ans}\n"
}

`mpf -c #{filename+"_LAMBDA_CV_PNORM2"} 2.0`
print "CV\n#{ans}\n"

