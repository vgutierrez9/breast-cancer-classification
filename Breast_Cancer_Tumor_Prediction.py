'''
Building Binary Classification Models to Identify Malignant and Benign
Breast Cancer Tumors

Vanessa Gutierrez
'''

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold


#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def prepare_dataset(dataset_path):
    '''
    Read a comma separated text file where
	- the first field is the sample code ID number
	- the 11th (last) field is a class label 2 for benign or 4 for malignant
	- the remaining fields are values for clump thickness, uniformity of cell size and
      shape, marginal adhesion, single epithelial cell size, bare nuclei, bland chromatin,
      normal nucleoli, and mitoses.

    Return two numpy arrays X and y where
	- X is two dimensional. X[i,:] is the ith tumor example and i's row contains the data points listed above for i
	- y is one dimensional. y[i] is the class label of tumor described at X[i,:]
          y[i] is set to 2 for benign or 4 for malignant

    @param dataset_path: full path of the dataset text file

    @return X,y
    '''
    
    # Read data from file, separated by commas
    X_All = np.genfromtxt(dataset_path, delimiter=",", dtype=int)
    # Convert data to numpyArray format
    X_All = np.array(X_All)

    X = list()
    Y = list()

    # Set Y to 2 (benign) or 4 (malignant) for each 2 or 4 in x[:][10]
    for elem in X_All:
        elem = list(elem)
        Y.append((elem[10],))
        # Remove ID and class from X
        X.append(elem[1:10])

    X = np.array(X)
    Y = np.array(Y)

    return(X, Y)
#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
def partition_dataset(X_All, Y_All):
    ''' Takes in the data set's X and Y, randomizes order of rows, 
    assigns 80% of data to Training and 20% to Testing, 
    and appropriate values to the respective training and testing 
    X and Y arrays.

    @param X_All an array with one tumor's data points per row, 
           with rows in the same order as the data was provided.
    @param Y_All an array with one tumor's classification, 2 for benign or 4 for malignant,
           per row, with each row's classification corresponding to the data in 
           the same row in X_All

    @return
    X_Train : randomized tumor records selected for the X_training set 
    X_Test : randomized tumor records selected for the X_testing set
    Y_Train : the respective label (2 or 4) corresponding to the records in X_Train
    Y_Test : the respective label (2 or 4) corresponding to the records in X_Test
    '''
    # Get the total number of records
    n = len(X_All)
    # Calculate the amount equal 80% of records, note the use of int.
    n80 = int(n*.8)

    # Create a list length n of randomized order of numbers 1-n
    randomOrder = np.random.permutation(n)
    
    # Randomize order of data, keeping X and Y in the same order as each other
    randomX = X_All[randomOrder]
    randomY = Y_All[randomOrder]

    # Separate data into training and testing sets 80:20
    X_Train = randomX[:n80]
    X_Test = randomX[n80:]
    Y_Train = randomY[:n80]
    Y_Test = randomY[n80:]

    return(X_Train, X_Test, Y_Train, Y_Test)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NB_classifier(X_training, y_training):
    '''
    Build a Naive Bayes classifier based on the training set X_training, y_training.

    @param
	X_training: X_training[i,:] where the ith row contains the data for one tumor
	y_training: y_training[i] is the class label of the tumor who's data is at X_training[i,:]

    @return
    best_clf : the classifier with the highest accuracy built in this function
    best_validation_accuracy : the classifier's validation accuracy
    CV_results : the results of cross validation
    '''

    CV_results = []
    kf = KFold(n_splits = 10)

    model = GaussianNB()
    
    # Use k-fold splits to train and validate 10 different Naive Bayes Classifiers
    for train, valid in kf.split(X_training):
        # Create a classifier using the training data from the current split
        clf = model.fit(X_training[train], y_training[train])
        # Adds the accuracy_score of validation data, and the classifier to the results list
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf),]

    best_validation_accuracy = 0
    
    for result in CV_results:
        # Saves the classifier with the best validation accuracy score from the k-fold validation
        if (result[0] > best_validation_accuracy):
            best_validation_accuracy = result[0]
            best_clf = result[1]

    return (best_clf, best_validation_accuracy, CV_results)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DT_classifier(X_training, y_training):
    '''
    Build a Decision Tree classifier based on the training set X_training, y_training.

   @param
	X_training: X_training[i,:] where the ith row contains the data for one tumor
	y_training: y_training[i] is the class label of the tumor who's data is at X_training[i,:]

    @return
    best_clf : the classifier with the highest accuracy built in this function
    best_validation_accuracy : the classifier's validation accuracy
    CV_results : the results of cross validation
    '''
    CV_results = []
    kf = KFold(n_splits = 10)

    model = DecisionTreeClassifier()
    
    # Use k-fold splits to train and validate 10 different Decision Tree Classifiers
    for train, valid in kf.split(X_training):
        clf = model.fit(X_training[train], y_training[train])
        # Adds the accuracy_score of validation data, and the classifier to the results list
        CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf),]

    best_validation_accuracy = 0
    
    for result in CV_results:
        # Saves the classifier with the best validation accuracy score from the k-fold validation
        if (result[0] > best_validation_accuracy):
            best_validation_accuracy = result[0]
            best_clf = result[1]

    return (best_clf, best_validation_accuracy, CV_results)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NN_classifier(X_training, y_training):
    '''
    Build a Nearest Neighbors classifier based on the training set X_training, y_training.

   @param
	X_training: X_training[i,:] where the ith row contains the data for one tumor
	y_training: y_training[i] is the class label of the tumor who's data is at X_training[i,:]

    @return
    best_clf : the classifier with the highest accuracy built in this function
    best_validation_accuracy : the classifier's validation accuracy
    CV_results : the results of cross validation
    '''
    kf = KFold(n_splits = 10)
    NN_results = []

    # Use cross validation to find best model for each odd value K in range 1-16
    for i in range(1,16):
        CV_results = []

        # Skip even k's for k nearest neighbor
        if (i % 2 != 0):
            model = KNeighborsClassifier(n_neighbors = i)
            
            # Use k-fold splits to train and validate 10 different kNN Classifiers for k = i value
            for train, valid in kf.split(X_training):
                clf = model.fit(X_training[train], y_training[train])
                # Adds the accuracy_score of validation data, and the classifier to the results list
                CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf),]

            best_accuracy_cv = 0
            # Saves the most accurate classifier for i neighbors (clf with highest validation accuracy)
            for result in CV_results:
                if(result[0] > best_accuracy_cv):
                    best_accuracy_cv = result[0]
                    best_clf_cv = result[1]

            # Save number of neighbors and the validation accuracy of the most accurate kNN classifier with i neighbors
            NN_results += [(i, best_accuracy_cv, best_clf_cv)]

    # Find the "best of the best": given the most accurate classifier for each k value, find and save the classifier with the best accuracy
    best_accuracy_K = 0
    for K in NN_results:
        if( K[1] > best_accuracy_K):
            best_accuracy_K = K[1]
            best_K = K[0]
            best_clf_K = K[2]

    return (best_clf_K, best_accuracy_K, best_K)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SVM_classifier(X_training, y_training):
    '''
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

   @param
	X_training: X_training[i,:] where the ith row contains the data for one tumor
	y_training: y_training[i] is the class label of the tumor who's data is at X_training[i,:]

    @return
    best_clf_K :  the classifier with the highest accuracy built in this function
    best_validation_accuracy_K : the classifier's validation accuracy
    best_K : the kernel used
    '''
    kf = KFold(n_splits = 10)
    SVM_results = []

    # Use cross validation to find best model for each kernel type
    for kernel in ('linear', 'rbf', 'sigmoid'):
        
        model = SVC(kernel=kernel)
        
        CV_results = []
        # Use k-fold splits to train and validate 10 different SVM Classifiers for each kernel type
        for train, valid in kf.split(X_training):
            clf = model.fit(X_training[train], y_training[train])
            # Adds the accuracy_score of validation data, and the classifier to the results list
            CV_results += [(accuracy_score(y_training[valid], clf.predict(X_training[valid])), clf),]

        best_validation_accuracy_cv = 0
        # Saves the most accurate classifier for the kernel type (clf with highest validation accuracy)
        for result in CV_results:
            if(result[0] > best_validation_accuracy_cv):
                best_validation_accuracy_cv = result[0]
                best_clf_cv = result[1]

        # Save kernel type and validation accuracy of the most accurate SVM classifier with that kernel type
        SVM_results += [(kernel, best_validation_accuracy_cv, best_clf_cv)]

    best_validation_K = 0
    # Find the "best of the best": given the most accurate classifier for each kernel type, find and save the classifier with the best accuracy
    for K in SVM_results:
        if(K[1] > best_validation_K):
            best_K = K[0]
            best_validation_K = K[1]
            best_clf_K = K[2]

    return (best_clf_K, best_validation_K, best_K)

#  - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":

    print("Breast Cancer Tumor Prediction")
    print("Author: Vanessa Gutierrez")
    print("This program builds various binary classifiers to predict the diagnoses of breast cancer tumors as malignant or benign, using the following tumor measurments:")
    print("Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape, Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, Bland Chromatin, Normal Nucleoli and Mitoses.")
    print("Using the Breast Cancer Wisconsin Dataset found at: \nhttp://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29\n")


    X, Y = prepare_dataset("Breast_Cancer_Wisconsin_Data.csv")
    X_Train, X_Test, Y_Train, Y_Test = partition_dataset(X, Y)

#     print(X_Train,X_Test, Y_Train, Y_Test)

    print("--------------------------------------------")
    print("NAIVE BAYES CLASSIFIER")
    # Create Naive Bayes classifier, and get its validation accuracy
    NB_clf, NB_validation_accuracy, NB_CV_results = build_NB_classifier(X_Train, Y_Train.ravel())
    print("Naive Bayes Classifier best accuracy on validation data in k-fold cross validation:", NB_validation_accuracy)
    # Run naive bayes classifier on testing data
    NB_testing_accuracy = accuracy_score(Y_Test, NB_clf.predict(X_Test))
    print("Naive Bayes Classifier Testing Data Accuracy: ", NB_testing_accuracy)

    print("--------------------------------------------")
    print("DECISION TREE CLASSIFIER")
    # Create Decision Tree classifier, and get its validation accuracy
    DT_clf, DT_validation_accuracy, DT_CV_results = build_DT_classifier(X_Train, Y_Train.ravel())
    print("Decision Tree Classifier best accuracy on validation data in k-fold cross validation:", DT_validation_accuracy)
    # Run Decision Tree classifier on testing data
    DT_testing_accuracy = accuracy_score(Y_Test, DT_clf.predict(X_Test))
    print("Decision Tree Classifier Testing Data Accuracy: ", DT_testing_accuracy)

    print("--------------------------------------------")
    print("K NEAREST NEIGHBORS (kNN) CLASSIFIER")
    # Create K-Nearest Neighbors classifier, and get its validation accuracy
    NN_clf, NN_validation_accuracy, NN_CV_results = build_NN_classifier(X_Train, Y_Train.ravel())
    print("K-Nearest Neighbor Classifier best accuracy on validation data in k-fold cross validation:", NN_validation_accuracy)
    # Run K-Nearest Neighbor classifier on testing data
    NN_testing_accuracy = accuracy_score(Y_Test, NN_clf.predict(X_Test))
    print("K-Nearest Neighbor Classifier Testing Data Accuracy: ", NN_testing_accuracy)

    print("--------------------------------------------")
    print("SUPPORT VECTOR MACHINE (SVM) CLASSIFIER")
    # Create SVM classifier, and get its validation accuracy
    SVM_clf, SVM_validation_accuracy, SVM_CV_results = build_SVM_classifier(X_Train, Y_Train.ravel())
    print("Support Vector Machine Classifier best accuracy on validation data in k-fold cross validation:", SVM_validation_accuracy)
    # Run SVM classifier on testing data
    SVM_testing_accuracy = accuracy_score(Y_Test, SVM_clf.predict(X_Test))
    print("Support Vector Machine Testing Data Accuracy: ", SVM_testing_accuracy)
    