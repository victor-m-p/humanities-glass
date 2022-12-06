load '../../ENT/ent.rb'

1.upto(10) { |nan|
  list=[]
  Dir.glob("NAN_TESTS/*NAN_20_#{nan}_*").collect { |f|
    file=File.open(f, 'r'); str=file.read; file.close
    ans=eval(str.scan(/^\[\[[^\n]+\n/)[-1])
  }.each { |i|
    list += i
  }

  print "#{nan} (#{list.length}): #{list.transpose[0].mean} #{list.transpose[1].mean} #{list.transpose[2].mean} #{list.transpose[3].mean}\n"
}
