from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    for i in range(len(train)):
        train[i] = train[i]/255
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]
    for i in range(28000):
        test[i,]=test[i,]/255

    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)

    # n_estimators is the number of trees in the forest
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(train, target)
    prediction = [[index + 1, x] for index, x in enumerate(rf.predict(test))]

    # info on Python formatting: http://docs.python.org/2/library/stdtypes.html#string-formatting-operations
    savetxt('Data/submission.csv', prediction, delimiter=',', fmt='%d,%d', 
            header='ImageId,Label', comments = '')

if __name__=="__main__":
    main()