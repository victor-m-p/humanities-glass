load "../../ENT/ent.rb"

class Hash
  def norm
    n=self.keys.collect { |i| self[i] }.sum
    self.keys.each { |i| self[i]=self[i]/n }
    self
  end
end

class Array
  def norm
    n=self.sum
    self.collect { |i| i/n }
  end
end

file=File.new("../data/analysis/configurations.txt", 'r')
file_p=File.new("../data/analysis/configuration_probabilities.txt", 'r')
pset=Hash.new()
file.each_line { |line|
  set=line.split(" ").collect { |i| i.to_i < 0 ? "0" : "1" }.join
  p=file_p.readline.to_f
  pset[set]=p
};1

list=[0,0,0,0]
(1 << 18).times { |pos|
  quote=pos.to_s(2)
  quote="0"*(18-quote.length)+quote
  ans=Array.new(4) { |loc|
    loc=loc.to_s(2)
    loc="0"*(2-loc.length)+loc
    final=quote[0..10]+loc+quote[11..-1]
    pset[final]
  }
  ans=ans.collect { |i| i/ans.sum}
  4.times { |i|
    list[i] += ans[i]
  }
}
list=list.collect { |i| i/list.sum}


ans=(Array.new(20) { |i| i }-base).collect { |flip|
  list_up=[0,0,0,0]
  list_down=[0,0,0,0]
  (1 << 18).times { |pos|
    quote=pos.to_s(2)
    quote="0"*(18-quote.length)+quote
    ans=Array.new(4) { |loc|
      loc=loc.to_s(2)
      loc="0"*(2-loc.length)+loc
      final=quote[0..10]+loc+quote[11..-1]
      pset[final]
    }
    ans=ans.collect { |i| i/ans.sum }
    if final[flip] == "1" then
      4.times { |i|
        list_up[i] += ans[i]
      }
    else
      4.times { |i|
        list_down[i] += ans[i]
      }      
    end
  }
  [flip, list_up.jsd(list_down), list_up.norm, list_down.norm]
}.sort { |i,j| j[1] <=> i[1] }

ans.collect { |i| i[0..1] }

ans[1]
