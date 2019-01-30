
data_name = 'blogcatalog'
# ot = open(data_name + '_part_info.txt', 'w')
f = open(data_name + '_edge.txt', 'r')
ls = f.readlines()
f.close()
f = open(data_name + '_label.txt', 'r')
ls2 = f.readlines()
f.close()
d = {}
s = set()
cnt = 0
cnt2 = 0
print("Edges: {}".format(len(ls)))
vs = set()
for l in ls:
    ids = l.strip().split(' ')
    vs.add(ids[0])
    vs.add(ids[1])
    pa = (ids[0], ids[1])
    if pa in s:
        cnt += 1
    s.add(pa)
    if ids[0] == ids[1]:
        continue
    pa = (ids[1], ids[0])
    if pa in s:
        cnt2 += 1
print("Nodes: {}".format(len(vs)))
print("Nodes according to labels: {}".format(len(ls2)))
print("double edge: {}".format(cnt))
print("reverse edge: {}".format(cnt2))
if cnt2 == 0:
    print("Undirected")
else:
    print("Directed")

lbs = set()
for l in ls2:
    xx = l.strip().split(' ')
    for lb in xx[1:]:
        lbs.add(lb)
print("Labels: {}".format(len(lbs)))

print(len(s))
exit()
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

vi = {}
for i in d.keys():
    vi[i] = 0
se = {}
cnt = 0

for i in vi.keys():
    if vi[i] == 0:
        vi[i] = 1
        q = [i]
        fp = 0
        while True:
            try:
                j = q[fp]
            except:
                break
            fp += 1
            for k in d[j]:
                if vi[k] == 0:
                    vi[k] = 1
                    q += [k]
        se[cnt] = fp
        cnt += 1


ot.writelines(['Number of parts: {}'.format(cnt), '\n'])
for i in se.keys():
    ot.writelines(['Part {}: {}'.format(i+1, se[i]), '\n'])
ot.close()
