with open('semantic_label_index.txt', 'r') as rf:
  label_list = []
  for line in rf.readlines():
    label_list.append(int(line.strip()))

indexs = set(label_list)
for i in indexs:
  nums = label_list.count(i)
  print(i, nums)

'''
results:

(1, 178)
(2, 8818)
(3, 2536)
(4, 6)
(5, 12187)
(6, 4762)
(7, 62719)
(8, 2999)
(9, 1956)
(10, 129)
(12, 114)
(13, 3574)
(14, 22)

'''
