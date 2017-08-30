with open('data/log_dataset.txt', 'r') as rf:
	with open('msg/_raw_msg.txt', 'w+') as wf:
		count = 0
		for line in rf.readlines():
			msg = line.strip('\n').split('\t')[5]
			wf.write(msg+'\n')
			count += 1
			print('saved %d messages !' % count)
			if count == 200000:
				break
			