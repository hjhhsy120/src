
import sys
filename = sys.argv[1]
pq = ['0.25', '0.5', '1.0', '2.0', '4.0']
micros = []
macros = []
for i in range(5):
    micros += [[]]
    macros += [[]]
    for j in range(5):
        micros[i] += [[]]
        macros[i] += [[]]
        f = open(filename + '_p' + pq[i] + '_q' + pq[j] + '.txt', 'r')
        for k in range(4):
            l = f.readline()
            ls = l.strip().split('\t')
            micros[i][j] += [ls[0]]
            macros[i][j] += [ls[1]]
        f.close()

titles = ['All', '<=20%', '20%~80%', '>80%']
ot = open(filename + '.txt', 'w')
ot.writelines(['p\tq'])
for j in range(5):
    ot.writelines(['\t', pq[j]])
ot.writelines('\n')
for i in range(5):
    for k in range(4):
        ot.writelines([pq[i], '\t', titles[k]])
        for j in range(5):
            ot.writelines(['\t', micros[i][j][k]])
        ot.writelines(['\n'])

ot.writelines(['\n\np\tq'])
for j in range(5):
    ot.writelines(['\t', pq[j]])
ot.writelines('\n')
for i in range(5):
    for k in range(4):
        ot.writelines([pq[i], '\t', titles[k]])
        for j in range(5):
            ot.writelines(['\t', macros[i][j][k]])
        ot.writelines(['\n'])
ot.close()

