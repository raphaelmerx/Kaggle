from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt(open('KaggleTutoData/train.csv','r'), delimiter=',', dtype='f8')[1:]    
    target = [x[0] for x in dataset]
    train = [x[1:] for x in dataset]
    test = genfromtxt(open('KaggleTutoData/test.csv','r'), delimiter=',', dtype='f8')[1:]

    #create and train the random forest
    #multi-core CPUs can use: rf = RandomForestClassifier(n_estimators=100, n_jobs=2)
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(train, target)
    predicted_probs = [[index + 1, x[1]] for index, x in enumerate(rf.predict_proba(test))]

    savetxt('KaggleTutoData/submission.csv', predicted_probs, delimiter=',', fmt='%d,%f',)

if __name__=="__main__":
    main()