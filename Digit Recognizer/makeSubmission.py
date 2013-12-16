from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('Data/train.csv','r'), delimiter=',', dtype='f8')[1:] 
    # each line x of the dataset is a digit, value and image
    # x[0] is the value of the digit 
    target = [x[0] for x in dataset]
    # x[1:] is a series of pixels
    train = [x[1:] for x in dataset]

    # test is the sample we want to predict, it contains only images, no values for the digits
    test = genfromtxt(open('Data/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    # n_estimators is the number of trees in the forest, a parameter for the Random Forest Algorithm
    rf = RandomForestClassifier(n_estimators=10)
    # the following line trains the model
    rf.fit(train, target)
    # rf.predict(test) is a list of all digit predictions for the file test.csv
    prediction = [[index + 1, x] for index, x in enumerate(rf.predict(test))]

    # info on Python formatting: http://docs.python.org/2/library/stdtypes.html#string-formatting-operations
    savetxt('Data/submission.csv', prediction, delimiter=',', fmt='%d,%d', 
            header='ImageId,Label', comments = '')

if __name__=="__main__":
    main()