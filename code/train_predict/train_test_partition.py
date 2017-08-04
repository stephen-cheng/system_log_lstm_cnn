
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
  # token  partition
  train_token, test_token, raw_token = 'msg_token_train.txt', 'msg_token_test.txt', 'msg_text_token_reshape.txt'
  train_test_part(train_token, test_token, raw_token)
  # label train_test partition
  train_token, test_token, raw_token = 'msg_label_train.txt', 'msg_label_test.txt', 'msg_label_reshape.txt'
  train_test_part(train_token, test_token, raw_token)
