

def msg_max_len():
  with open('data/log_dataset.txt', 'r') as rf:
    max_len = 0
    for line in rf.readlines():
      msg = line.split('\t')[5].strip().split()
      line_len = len(msg)
      if line_len > max_len:
        max_len = line_len
  return max_len

def msg_label():
  with open('data/log_dataset.txt', 'r') as rf:
    label_name = []
    for line in rf.readlines():
      msg_name = line.split('\t')[4].strip()
      if msg_name not in label_name:
        label_name.append(msg_name)
  return label_name

def msg_classify(label_name):
  for label in label_name:
    with open('data/log_dataset.txt', 'r') as rf:
      with open('msg/'+label+'.txt', 'w+') as wf:
        for line in rf.readlines():
          if line.split('\t')[4].strip() == label:
            wf.write(line.split('\t')[5].strip()+'\n')
    print('log message ' + label + ' has been generated !')
	
def msgX128(label_name):
  for label in label_name:
    msg_text = []
    with open('msg/'+label+'.txt', 'r') as rf:
      for line in rf.readlines():
        msg_text.extend(line.strip().split())
    time_steps = len(msg_text) // 128
    with open('msgX128/'+label+'X128.txt', 'w+') as wf:
      for i in range(time_steps):
        start = i * 128
        end = (i+1) * 128
        line = ' '.join(msg_text[start:end])
        wf.write(line+'\n')
 
def train_test(label_name):	
  for label in label_name:
    with open('msgX128/'+label+'X128.txt', 'r') as rf:
      with open('msgX128/msg_text.txt', 'a+') as wf_text:
        with open('msgX128/msg_label.txt', 'a+') as wf_label:
          for line in rf.readlines():
            wf_text.write(line)
            wf_label.write(str(label_name.index(label))+'\n')
			
def train_test_len():
  with open('msgX128/msg_text.txt', 'r') as rf:
    i = 0
    for line in rf.readlines():
      i += 1
  print('msg len: ', i)
  with open('msgX128/msg_label.txt', 'r') as rf:
    i = 0
    for line in rf.readlines():
      i += 1
  print('label len: ', i)
                 
if __name__ == '__main__':
  # labels = ['info', 'crit', 'err', 'notice', 'warning', 'alert', 'emerg']
  label_name = msg_label()
  #msg_classify(label_name)
  #msgX128(label_name)
  #label_num = zip(label_name, range(len(label_name)))
  train_test(label_name)
  train_test_len()