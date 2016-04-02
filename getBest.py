
import sys

scores = []

while True:
    line = sys.stdin.readline()
    if not line:
        break
    if line.startswith('Iter '):
        break

while True:
    line = sys.stdin.readline()
    if not line:
        break
    array = line.strip().split('\t')
    if len(array) != 3:
        continue
    if float(array[1]) != 1951:
        continue
    scores.append(float(array[2]))

sort_scores = sorted(scores, reverse=True)

for s in sort_scores:
    print s
    break



