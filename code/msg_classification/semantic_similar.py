from __future__ import print_function
import os
from leven_distance import levenshtein_distance as ld

# read log message

def read_file(filename):
  with open(filename, 'r') as rf:
    context = []
    for line in rf:
      line_str = line.strip()
      context.append(line_str)
  return context

filename = 'data/raw_msg.txt'
context = read_file(filename)


# build semantic dict

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
  label_dict = seman_dict.copy()
  
  # create dict
  for label in label_list:
    with open(dir+'/'+label+'.txt', 'r') as f:
      keywords = []
      for word in f:
        keywords.append(word.strip())
      seman_dict[label] = keywords	  
  return label_dict, seman_dict

dir = 'message_type'
label_dict, semantic_dict = semantic_dict(dir)
print('semantic label: ', label_dict)


#semantic similarity

def semantic_sim(line_str, label_dict, semantic_dict):
  label_dict_sum = label_dict.copy()
  for key in label_dict.keys():
    label_dict[key] = []  
    for word in semantic_dict[key]:
      leven_sim = ld(line_str, word).leven_sim() * 10.0
      label_dict[key].append(leven_sim)
    label_dict[key] = sorted(label_dict[key])[-10:]
    label_dict[key].reverse()
    label_dict_sum[key] = sum(label_dict[key])
    #print(label_dict_sum)
  max_count = max(label_dict_sum.values())
  semantic_label = [k for (k, v) in label_dict_sum.items() if v == max_count][0]
  return label_dict, semantic_label

with open('data/semantic_label.txt', 'w+') as lf:
  with open('data/semantic_sim.txt', 'w+') as sf:
    for line_str in context:
      label_dict, semantic_label = semantic_sim(line_str, label_dict, semantic_dict)
      lf.write(semantic_label + '\n')
      sim_list = []
      for sim in label_dict.values():
        sim_list.extend(sim)		
      label_count = '\t'.join([str(i) for i in sim_list])
      sf.write(label_count + '\n')
      print('semantic classification label: ', semantic_label)
    print(label_dict)
    '''
	{'file': [], 'network': [], 'service': [], 'database': [], 'communication': [], 
	 'memory': [], 'driver': [], 'system': [], 'application': [], 'io': [], 
	 'others': [], 'security': [], 'disk': [], 'processor': []}
	'''

