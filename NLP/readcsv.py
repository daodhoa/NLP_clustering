import os, csv
from preprocessing import preprocess

f=open("other.csv",'a+')
w=csv.writer(f)

directory = '/home/duyhoa/PycharmProjects/NLP/20news-bydate-train/'

with open('continue.txt', 'r') as file:
    filenames = file.readlines()

for filename in filenames:
    with open(directory + filename.strip(), 'rb') as f1:
        print("--reading file: " + filename.strip())
        sample = f1.read()
        # try:
        #     sample = f1.read()
        # except:
        #     print("can't read file: " + filename.strip())
    try:
        output = preprocess(str(sample))
        w.writerow([filename.strip()[-5:], output, filename.strip()[:-6]])
    except:
        print("Can not preprocess file: " + filename.strip())
f.close()