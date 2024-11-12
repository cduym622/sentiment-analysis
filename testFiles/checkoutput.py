from os import listdir, getcwd
from os.path import isfile, join
path = getcwd()

onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
print(onlyfiles)
def left(s, amount):
    return s[:amount]

def right(s, amount):
    return s[-amount:]
import csv


for f in onlyfiles:
    count = 0
    if(right(f,4)=='.txt' or right(f,4)=='.csv'):
        print('checking '+f)
        with open(f) as csv_file:
          csv_reader = csv.reader(csv_file, delimiter='\t')
          for row in csv_reader:
            if (len(row)!=2):
                print('error: expecting 2 fields that are tab delimited')
            id = row[0].strip()
            ans = row[1].strip()
            if id.isnumeric(): #ignore header if it exists
                count+=1

if (count < 99990):
    print('error: expected at least 99990 rows')
if (ans != '+' and ans != '-'):
    print('error: each entry should have + or - in the second column')