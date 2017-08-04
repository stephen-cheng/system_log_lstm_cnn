
def msg_window_resize():
  with open('msgX128/msg_text_token_reshape.txt', 'w+') as wf:
    with open('msgX128/msg_text_token.txt', 'r') as rf:
      count = 0
      drop_line = []
      for line in rf.readlines():
        msg_line = line.strip().split()
        count +=1 
        if len(msg_line) < 120:
          drop_line.append(count)
        else:
          wf.write('\t'.join(msg_line[:120])+'\n')
  return drop_line

def label_resize(drop_line):
  with open('msgX128/msg_label_reshape.txt', 'w+') as wf:
    with open('msgX128/msg_label.txt', 'r') as rf:
      count = 0
      for line in rf.readlines():
        label = line.strip()
        count +=1 
        if count not in drop_line:
          wf.write(label+'\n')

if __name__ == '__main__':
  drop_line = msg_window_resize()
  print('msg text token\'s reshaped !')
  print('The size of lines whose window is less than 128: ', len(drop_line))
  label_resize(drop_line)
  print('Label resized successfully !')
