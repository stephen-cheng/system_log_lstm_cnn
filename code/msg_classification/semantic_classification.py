from __future__ import print_function
import os

def read_file(filename):
  with open(filename, 'r') as rf:
    word_list = []
    for line in rf:
      word_list.append(line.strip().split(' '))
  return word_list
  
def sep_name(fname):
	label=fname.split(".")[0]
	return label
  
def semantic_dict(dir):
  # build dict key
  if os.path.isdir(dir):
    filelist = os.listdir(dir)
    label_list = []
    for filename in filelist:
      label = sep_name(filename)
      label_list.append(label)
  else:
    print('File directory is not existed !')
  seman_dict = dict.fromkeys(label_list)
  dict_label = seman_dict.copy()
  
  # create dict
  for label in label_list:
    with open(dir+'/'+label+'.txt', 'r') as f:
      keywords = []
      for word in f:
        keywords.append(word.strip())
      seman_dict[label] = keywords	  
  return dict_label, seman_dict

def keyword_count(word_list, dict_label, semantic_dict):
  for key in dict_label.keys():
    dict_label[key] = 0
    for word in word_list:  
      if word in semantic_dict[key]:
        dict_label[key] += 1
      else:
        pass
  max_count = max(dict_label.values())
  classify_label = [k for (k, v) in dict_label.items() if v == max_count][0]
  if max_count == 0:
    classify_label = 'others'
  return dict_label, classify_label
  
  
# read filtered log file
filename = 'data/log_msg_filtered.txt'
words = read_file(filename)
#print('log sample: ', words[:9])

# build semantic dict
dir = 'message_type'
dict_label, semantic_label = semantic_dict(dir)
print('semantic label: ', dict_label)

# keywords count
with open('data/log_msg_label.txt', 'w+') as f:
  with open('data/type_count.txt', 'w+') as lf:
    for word_list in words:
      label_dict, classify_label = keyword_count(word_list, dict_label, semantic_label)
      f.write(classify_label + '\n')
      label_count = '\t'.join([str(i) for i in label_dict.values()])
      lf.write(label_count + '\n')
      print('semantic classification label: ', classify_label)
    print(label_dict)
    '''
    {'file': 0, 'network': 0, 'service': 0, 'database': 0, 
	 'communication': 0, 'memory': 0, 'driver': 0, 'system': 0, 
	 'application': 3, 'io': 0, 'others': 0, 'security': 0, 
	 'disk': 0, 'processor': 0}
    '''



