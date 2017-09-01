label_names =['file', 'network', 'service', 'database', 'communication', 'memory', 'driver', 
			  'system', 'application', 'io', 'others', 'security', 'disk', 'processor']

with open('data/semantic_label.txt', 'r') as rf:
  with open('data/semantic_label_index.txt', 'w+') as wf:
    count = 0
    for line in rf.readlines():
      label = line.strip('\n')
      label_index = label_names.index(label)+1
      wf.write(str(label_index)+'\n')
      count += 1
      if count % 1000 == 0:
        print('saved %d lables !' % count)
    print('Finished!')
    