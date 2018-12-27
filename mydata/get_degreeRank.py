
data_name = 'email'
ot = open(data_name + '_degreeRank.txt', 'w')
f = open(data_name + '_edge.txt', 'r')
ls = f.readlines()
f.close()
d = {}
degrees = {}
for l in ls:
    ids = l.split('\n')[0].split(' ')
    if ids[0] in d.keys():
        if not ids[1] in d[ids[0]]:
            d[ids[0]] += [ids[1]]
    else:
        d[ids[0]] = [ids[1]]
    if ids[1] in d.keys():
        if not ids[0] in d[ids[1]]:
            d[ids[1]] += [ids[0]]
    else:
        d[ids[1]] = [ids[0]]
for i in d.keys():
    degrees[i] = len(d[i])

nodenum = len(d.keys())
st = sorted(degrees.items(), key=lambda x: x[1])
res = {}
for i in range(nodenum):
    res[st[i][0]] = i
for i in res.keys():
    ot.writelines([i, ' ', str(res[i]), '\n'])
ot.close()
