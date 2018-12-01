from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



def SVM_cross_validation(n_components):

    max_score = -999
    
    # #############################################################################
    # Load the data as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=0.4, data_home='data', download_if_missing=False)
    #print("training model's people names: " + str(lfw_people.target_names))
    #print("Training using people images: " + str(lfw_people.data))

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    print("training model's people names: " + str(lfw_people.target_names))

    # #############################################################################
    # Split into a training set and a test set using a stratified k fold
    iteration = 0
    kf = KFold(n_splits=8, shuffle=True, random_state=42) #=5

    for train_index, test_index in kf.split(X):
        iteration += 1
        print('\nKFold iteration time: ' + str(iteration))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        '''
        # split into a training and testing set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42)
        '''

        # #############################################################################
        # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
        # dataset): unsupervised feature extraction / dimensionality reduction
        #n_components = .6 ###change value for test
        n_components_pct = n_components*100

        print("Extracting the top %d%% eigenfaces from %d faces"
              % (n_components_pct, X_train.shape[0]))
        t0 = time()##################
        #pca = PCA(n_components=n_components, svd_solver='randomized',
                  #whiten=True).fit(X_train)
        pca = PCA(n_components=n_components,
                  whiten=True).fit(X_train)
        #print("done in %0.3fs" % (time() - t0))
        
        print("Projecting the input data on the eigenfaces orthonormal basis")
        #t0 = time()##################
        X_train_pca = pca.transform(X_train)
        X_test_pca = pca.transform(X_test)
        #print("done in %0.3fs" % (time() - t0))


        # #############################################################################
        # Train a SVM classification model

        print("Fitting the classifier to the training set")
        #t0 = time()##################
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                           param_grid, cv=5)
        clf = clf.fit(X_train_pca, y_train)
        #print("done in %0.3fs" % (time() - t0))
        #print("Best estimator found by grid search:")
        #print(clf.best_estimator_)


        # #############################################################################
        # Quantitative evaluation of the model quality on the test set

        print("Predicting people's names on the test set")
        #t0 = time()##################
        y_pred = clf.predict(X_test_pca)
        score = clf.score(X_test_pca, y_test)

        if score > max_score:
            max_score = score
        
        print("Score is: " + str(score))
        #print(clf)
        print("done in %0.3fs" % (time() - t0))

        #print(classification_report(y_test, y_pred, target_names=target_names))
        #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

        #if you want to plot the final result, uncommand following 2 lines
        #prediction_titles = [title(y_pred, y_test, target_names, i)                             for i in range(y_pred.shape[0])]
        #plot_gallery(X_test, prediction_titles, h, w)
        #plt.show()

    print("\nMaxmium SVM with corss validation score is: " + str(max_score))


###################################################################
def SVM_no_cross_validation(n_components):

    max_score = -999
    lfw_people = fetch_lfw_people(min_faces_per_person=1, resize=0.4, data_home='data', download_if_missing=False)
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    n_features = X.shape[1]
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    #print("training model's people names: " + str(lfw_people.target_names))

    #######################################
    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)
    
    n_components_pct = n_components*100

    print("Extracting the top %d%% eigenfaces from %d faces"
          % (n_components_pct, X_train.shape[0]))
    t0 = time()##################
    
    pca = PCA(n_components=n_components,
              whiten=True).fit(X_train)
    
    print("Projecting the input data on the eigenfaces orthonormal basis")
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Train a SVM classification model
    print("Fitting the classifier to the training set")

    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }

    #param_grid = {'C': [1e3],
                  #'gamma': [0.00001], }
    
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),                      param_grid, cv=5)
    clf = clf.fit(X_train_pca, y_train)
    # Quantitative evaluation of the model quality on the test set
    print("Predicting people's names on the test set")
    
    y_pred = clf.predict(X_test_pca)
    score = clf.score(X_test_pca, y_test)

    if score > max_score:
        max_score = score

    print("\nSVM without corss validation score is: " + str(max_score))
    print("done in %0.3fs" % (time() - t0))

    #if you want to plot the final result, uncommand following 2 lines
    prediction_titles = [title(y_pred, y_test, target_names, i)                             for i in range(y_pred.shape[0])]
    plot_gallery(X_test, prediction_titles, h, w)
    plt.show()


# #############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set
def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    if pred_name != 'Bush':
        pred_name = 'Not Bush'
    return 'Predicted: %s\nName:      %s' % (pred_name, true_name)


def main():
    print(__doc__)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    #SVM_cross_validation(.8) #choose the amount of eigenfaces
    SVM_no_cross_validation(.8) #choose the amount of eigenfaces
    

if __name__ == '__main__':
    main()

