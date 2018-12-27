f = open('amazon_label.txt', 'r')
for i in range(20):
  f.readline()

f.close()
f = open('com-amazon.all.dedup.cmty.txt', 'r')
for i in range(20):
  f.readline()

f.close()