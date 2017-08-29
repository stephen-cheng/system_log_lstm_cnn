import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def text_tokenize():
  # tokenize and count the word occurrences of a minimalistic corpus of text documents:
  vectorizer = CountVectorizer(min_df=1)
  analyze = vectorizer.build_analyzer()
  with open('msgX128/msg_text.txt', 'r') as rf:
    corpus, corpus_list = [], []
    for line in rf.readlines():
      line = line.decode('utf-8', 'ignore')
      corpus.append(line.strip())
      corpus_list.append(analyze(line.strip()))
  X = vectorizer.fit_transform(corpus)
  token = vectorizer.get_feature_names()
  #print token
  words_token = X.toarray()
  
  '''
  # max token_sum
  words_sum = words_token.sum(axis=0)
  token_sort = sorted(words_sum, reverse=True)
  print token_sort[:128]
  
  # max token_sum index
  token_index_sort = np.argsort(-words_sum)
  token_index = token_index_sort[:128]
  #print token_index
  '''

  # token index
  token_window = []
  for msg in corpus_list:
    token_window.append([token.index(i) for i in msg])	
  
  # save tokenized msg array
  with open('msgX128/msg_text_token.txt', 'w+') as wf:
    for tw in token_window:
      msg_window = ' '.join([str(i) for i in tw])
      wf.write(msg_window+'\n')
  print('Tokenized msg text file generated successfully !')

  return words_token, token_window

if __name__ == '__main__':
  token_array, token_windows = text_tokenize()
  # shape = 55829 * X
  print('msg token shape: ', token_array.shape)
  print('msg taken window size: ', len(token_windows), len(token_windows[0]))
