from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt, array, mean
import csv


def fileToArray(filename, columnsToKeep):
    reader = csv.reader(open(filename,'r'))
    lines = []
    header = reader.next()
    toKeepIds = [header.index(x) for x in columnsToKeep]
    for line in reader:
        line[:] = [ item for i,item in enumerate(line) if i in toKeepIds ]
        for i in range(len(line)):
            if line[i]=='male':
                line[i] = -1
            elif line[i]=='female':
                line[i]=1
            elif line[i].isdigit():
                line[i] = int(line[i])
        #line[:] = [-1 if x=='male' else x for x in line]
        #line[:] = [1 if x=='female' else x for x in line]
        lines.append(line)
    return array(lines)

# TODO: add column age
dataset = fileToArray('Data/train.csv', columnsToKeep = ['Survived','Pclass','Sex','Age', 
    'SibSp', 'Parch'])
test = fileToArray('Data/test.csv',columnsToKeep = ['Pclass','Sex','Age','SibSp', 'Parch'])
#create the training & test sets, skipping the header row with [1:]

# function that takes an array ages, and fills missing values with the mean
def missingToMean(ages):
    listWithoutMissing = []
    for age in ages:
        if age != '':
            listWithoutMissing.append(age.astype(float))
    agesMean = mean(listWithoutMissing)
    for i in range(ages.size):
        if ages[i]=='':
            ages[i] = agesMean
    return ages

# filling the missing ages with the mean of the ages
dataset[:,3] = missingToMean(dataset[:,3])
dataset = dataset.astype(float).astype(int)
test[:,2] = missingToMean(test[:,2])
test = test.astype(float).astype(int)

target = [int(x[0]) for x in dataset]
train = [x[1:] for x in dataset]
rf = RandomForestClassifier(n_estimators=10)
rf.fit(train, target)
prediction = [[index + 892, x] for index, x in enumerate(rf.predict(test))]

# info on Python formatting: http://docs.python.org/2/library/stdtypes.html#string-formatting-operations
savetxt('Data/submission.csv', prediction, delimiter=',', fmt='%d,%d', 
        header='PassengerId,Survived', comments = '')