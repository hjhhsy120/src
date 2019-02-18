
data_name = 'email_con'
ot = open(data_name + '_info2.txt', 'w')
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
pairs = {}
for i in d.keys():
    x = len(d[i])
    degrees[i] = x
    for iii in range(x - 1):
        for jj in d[i][iii + 1:]:
            ii = d[i][iii]
            if not ii in pairs.keys():
                y = {}
                y[jj] = 1
                pairs[ii] = y
            else:
                if not jj in pairs[ii].keys():
                    pairs[ii][jj] = 1
                else:
                    pairs[ii][jj] += 1
print('pairs built')
nodenum = len(d.keys())
totdegree = 0
for i in degrees.keys():
    totdegree += degrees[i]
ot.writelines(['Average degree: {}\n\n'.format(totdegree / nodenum)])

degrees = sorted(degrees.items(), key=lambda x: x[1])
ot.writelines(['degree distribution:\n'])
for i in range(9):
    ot.writelines(['{}% : {}\n'.format((i + 1) * 10, degrees[int(nodenum * (i + 1) / 10)][1])])
ot.writelines(['\n'])
print('degree finished')
f = open(data_name + '_label.txt', 'r')
ls = f.readlines()
f.close()
ld = {}
for l in ls:
    ids = l.split('\n')[0].split(' ')
    for j in ids[1:]:
        if j in ld.keys():
            ld[j] += [ids[0]]
        else:
            ld[j] = [ids[0]]

# common neighbor cnt & same label pair cnt
commcnt = 0
totcnt = 0
for k in ld.keys():
    x = len(ld[k])
    if x == 1:
        continue
    for ii in range(x - 1):
        for j in ld[k][ii + 1:]:
            totcnt += 1
            try:
                commcnt += pairs[ld[k][ii]][j]
            except:
                pass
            try:
                commcnt += pairs[j][ld[k][ii]]
            except:
                pass

ot.writelines(['Average number of common neighbor between each label\'s node pairs: {}\n'.format(commcnt / totcnt)])
ot.writelines(['Number of pairs: {}\n'.format(totcnt)])
ot.close()
