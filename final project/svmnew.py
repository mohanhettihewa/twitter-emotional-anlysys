import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report

def usage():
    print("Usage:")
    print("python %s <data_dir>" % sys.argv[0])

def read_data(fname):
    data = []
    f = open(fname, 'r')
    line = f.readline()
    while line != '':
        data.append([line])
        line = f.readline()
    f.close()
    return data    

if __name__ == '__main__':

    

    data_dir = "c:/Users/MaduZ/Desktop/final project/data/"
    
    classes =  ['joy','anger','sadness','love','surprise','fear']
    #train_labels =['joy','anger','sadness','love','surprise','fear']
    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                 
                content = f.read()
                #content = [line.strip() for line in f]
                
                #test_data.append(read_data(content))
                test_labels.append(curr_class)
                
                train_data.append(content)
                train_labels.append(curr_class)
                #test_data.append(content)

    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=1,
                                 max_df = 0.8,
                                 sublinear_tf=True,
                                 use_idf=True,
                                 ) 
   # train_data = ['cool', 'wow', 'briliant', 'awesome']
    train_vectors = vectorizer.fit_transform(train_data)
    x=0
    while(x < 100):
          f = open('Texxt%s.txt'%x,'r')
          tcontent = f.read() 
          test_data.append(tcontent) 
          x=x+1

    #test_data=[line.strip() for line in open('test1.txt')]
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1

    # Perform classification with SVM, kernel=linear
    # classifier_linear = svm.SVC(kernel='linear')
    # t0 = time.time()
    # classifier_linear.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_linear = classifier_linear.predict(test_vectors)
    # t2 = time.time()
    # time_linear_train = t1-t0
    # time_linear_predict = t2-t1 

    # Perform classification with SVM, kernel=linear
    # classifier_liblinear = svm.LinearSVC()
    # t0 = time.time()
    # classifier_liblinear.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_liblinear = classifier_liblinear.predict(test_vectors)
    # t2 = time.time()
    # time_liblinear_train = t1-t0
    # time_liblinear_predict = t2-t1 

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))
