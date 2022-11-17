#!/usr/bin/ruby
$stdout.sync = true

n_proc=32
fact=16
require 'parallel'

class Array
  def mean
    self.sum/(1.0*self.length)
  end
  
  def var
    avg=self.mean
    
    self.inject(0) { |num, i| num + (i-avg)**2 }/self.length
  end  
end

[[100, 10], [200, 10], [200, 20], [400, 20], [800, 20], [200, 40], [400, 40], [800, 40]].each { |pair|
  [0.1, 0.5, 1.0, 2.0].each { |beta|
    [1,2,3].each { |nn|

      print "Starting #{pair} / #{beta} / #{nn} at #{Time.now}\n"
      start=Time.now

      ans5=Parallel.map(Array.new(fact*n_proc) {}, in_processes: n_proc) { str=`./mpf -s #{pair.join(" ")} #{beta} 1000 0.0 #{nn}` }.collect { |i| [i.scan(/true:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/true:[^\n]+\n/)[1].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[1].split(" ")[-1].to_f] }

      print "Round done (#{(Time.now-start)/60} minutes)\n"
      [ans5].each { |ansN|
        print "#{ansN.collect { |i| i[0]-i[2] }.mean} ± #{(ansN.collect { |i| i[0]-i[2] }.var/ansN.length)**0.5}\n#{ansN.collect { |i| i[1]-i[3] }.mean} ± #{(ansN.collect { |i| i[1]-i[3] }.var/ansN.length)**0.5}\n"
        print "\n"
      }

      ans7=Parallel.map(Array.new(fact*n_proc) {}, in_processes: n_proc) { str=`./mpf -s #{pair.join(" ")} #{beta} 1000 10.0 #{nn}` }.collect { |i| [i.scan(/true:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/true:[^\n]+\n/)[1].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[1].split(" ")[-1].to_f] }

      print "Round done (#{(Time.now-start)/60/3.0} minutes)\n"
      [ans7].each { |ansN|
        print "#{ansN.collect { |i| i[0]-i[2] }.mean} ± #{(ansN.collect { |i| i[0]-i[2] }.var/ansN.length)**0.5}\n#{ansN.collect { |i| i[1]-i[3] }.mean} ± #{(ansN.collect { |i| i[1]-i[3] }.var/ansN.length)**0.5}\n"
        print "\n"
      }

      ans9=Parallel.map(Array.new(fact*n_proc) {}, in_processes: n_proc) { str=`./mpf -s #{pair.join(" ")} #{beta} 1000 100.0 #{nn}` }.collect { |i| [i.scan(/true:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/true:[^\n]+\n/)[1].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[0].split(" ")[-1].to_f,
        i.scan(/inferred:[^\n]+\n/)[1].split(" ")[-1].to_f] }

      print "Round done (#{(Time.now-start)/60/3.0} minutes)\n"
      [ans9].each { |ansN|
        print "#{ansN.collect { |i| i[0]-i[2] }.mean} ± #{(ansN.collect { |i| i[0]-i[2] }.var/ansN.length)**0.5}\n#{ansN.collect { |i| i[1]-i[3] }.mean} ± #{(ansN.collect { |i| i[1]-i[3] }.var/ansN.length)**0.5}\n"
        print "\n"
      }
      
    }
  }
}
