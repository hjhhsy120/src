
data_name = 'email'
ot = open(data_name + '_con_edge.txt', 'w')
otl = open(data_name + '_con_label.txt', 'w')
f = open(data_name + '_edge.txt', 'r')
fl = open(data_name + '_label.txt', 'r')
ls = f.readlines()
f.close()

all = set()
father = {}

def findset(x):
    global father
    if x == father[x]:
        return x
    father[x] = findset(father[x])
    return father[x]

def myunion(x, y):
    global father
    u = findset(x)
    v = findset(y)
    father[v] = u


for l in ls:
    ids = l.strip().split(' ')
    if not ids[0] in all:
        all.add(ids[0])
        father[ids[0]] = ids[0]
    if not ids[1] in all:
        all.add(ids[1])
        father[ids[1]] = ids[1]
    myunion(ids[0], ids[1])

d = {}
for i in all:
    try:
        d[findset(i)].add(i)
    except:
        d[findset(i)] = {i}
j = 0
k = 1
for i in d.keys():
    t = len(d[i])
    if t > j:
        j = t
        k = i
s = set()
for l in ls:
    ids = l.strip().split(' ')
    if ids[0] != ids[1] and ids[0] in d[k] and ids[1] in d[k]:
        if not (ids[0], ids[1]) in s:
            s.add((ids[0], ids[1]))
            ot.writelines([l])
ot.close()

for l in fl.readlines():
    ids = l.strip().split(' ')
    if ids[0] in d[k]:
        otl.writelines([l])
otl.close()
fl.close()

