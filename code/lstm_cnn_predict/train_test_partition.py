
def data_even(raw_file, new_file):
  with open('data/'+raw_file, 'r') as rf:
    with open('data/'+new_file, 'w+') as wf:
      line_num = 0
      for line in rf.readlines():
        line_num += 1
        if line_num <= 10000 or line_num >=53218:
          wf.write(line.strip()+'\n')

def train_test_part(train_file, test_file, raw_file):
  with open('data/'+train_file, 'w+') as wf_train:
    with open('data/'+test_file, 'w+') as wf_test:
      with open('data/'+raw_file, 'r') as rf:
        line_num = 0
        for line in rf.readlines():
          line_num += 1
          if line_num % 6 == 0:
            wf_test.write(line.strip()+'\n')
          else:
            wf_train.write(line.strip()+'\n')

if __name__ == '__main__':
  # even data distribution
  raw_file, new_file = 'msg_text_token_reshape.txt', 'msg_text_token_new.txt'
  data_even(raw_file, new_file)
  raw_file, new_file = 'msg_label_reshape.txt', 'msg_label_new.txt'
  data_even(raw_file, new_file)
  
  # token partition
  train_token, test_token, raw_token = 'msg_token_train.txt', 'msg_token_test.txt', 'msg_text_token_new.txt'
  train_test_part(train_token, test_token, raw_token)
  # label train_test partition
  train_token, test_token, raw_token = 'msg_label_train.txt', 'msg_label_test.txt', 'msg_label_new.txt'
  train_test_part(train_token, test_token, raw_token)
