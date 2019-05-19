import os, csv
from preprocessing import preprocess

f=open("data.csv",'a+')
w=csv.writer(f)

directory = '/home/duyhoa/PycharmProjects/NLP/20news-bydate-train/'

def convert_csv(dir):
    folder = directory + dir+'/'
    for path, dirs, files in os.walk(folder):
        for filename in files:
            with open(folder + filename, 'r') as f:
                print("--reading file: " + dir + "/" + filename)
                try:
                    sample = f.read()
                except:
                    print("can't read file: " + dir + "/" + filename)
            try:
                output = preprocess(sample)
                w.writerow([filename, output, dir])
            except:
                print("Can not preprocess file: " + filename + "folder: " + dir)
    f.close()

if __name__ == '__main__':
    for path, dirs, files in os.walk(directory):
        for folder_name in dirs:
            print("-------read folder: " + folder_name)
            convert_csv(folder_name)
