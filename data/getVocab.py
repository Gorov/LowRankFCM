import sys

files = ['wsj.train.nextpos', 'wsj.test.nextpos']

vocab = {}

for filename in files:
	filein = open(filename, 'r')
	while True:
		line = filein.readline()
		if not line:
			break
		array = line.strip().split('\t')
		num = int(array[2])
		for word in array[:2]:
			if word not in vocab:
				vocab[word] = 1
				print word
		for i in range(num):
			line = filein.readline().strip()
			array = line.split('\t')
			if array[1] not in vocab:
				vocab[array[1]] = 1
				print array[1]
		#sys.stdin.readline()
	filein.close()
