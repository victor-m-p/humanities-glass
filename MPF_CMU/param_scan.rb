require 'parallel'

n_nodes=20
n_obs=100
beta=0.25

`./mpf -g test #{n_nodes} #{n_obs} #{beta}`
scan=[-100]+Array.new(101) { |i| (i-50.0)/25 }
kl_fit_1=Parallel.map(scan, :in_process=>8) { |logs|
  ans=[logs, `./mpf -t test #{logs} 1`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
  print "#{ans}\n"
  ans
}
kl_fit_2=Parallel.map(scan, :in_process=>8) { |logs|
  ans=[logs, `./mpf -t test #{logs} 2`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
  print "#{ans}\n"
  ans
}
kl_fit_3=Parallel.map(scan, :in_process=>8) { |logs|
  ans=[logs, `./mpf -t test #{logs} 3`.scan(/ergence:[^\n]+\n/)[0].split(":")[-1].to_f]
  print "#{ans}\n"
  ans
}

