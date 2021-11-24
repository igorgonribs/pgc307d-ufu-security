import time
import sys

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

def visualize_predictions(results):
    ## Visualization of the predicted values
    ## DDoS predictions
    ddos = tuple(filter(lambda x: x == 'DDoS', results))
    print("Number of DDoS predictions: %2d" % len(ddos))

    ## Benign predictions
    benign = tuple(filter(lambda x: x == 'BENIGN', results))
    print("Number of Benign predictions: %2d" % len(benign))


def calculate_hit_rate(test_target, results):
    hits = (results == test_target.values)

    hit_rate = 100.0 * sum(hits)/len(test_target)
    print("Our algorithm hit rate: %f" % hit_rate)


def predict_values(test_data, model):
    print('Starting prediction...')
    results = model.predict(test_data)
    print('Prediction complete...')
    print()
    return results

def train_model(training_data, training_target, model):
    print('Starting training...')
    start_time = time.time()
    model.fit(training_data, training_target.values.ravel())
    end_time = time.time()
    print('Training complete...')
    print('Duration of training: ', (end_time-start_time), 'seconds.')
    print()
    return model

def set_trainingset_and_testset(y, x, rate_training, test_training):
    print('Setting training and test data...')
    training_size = rate_training*len(y)
    test_size = test_training*len(y)
    print('The training set contains %2d registers' % training_size)
    print('The testing set contains %2d registers' % test_size)
    print()
    training_data, test_data, training_target,test_target = train_test_split(x, y, test_size=test_training, random_state=42)
    return training_data,training_target,test_data,test_target

def normalize_features(x):
    print('Normalizing data...')
    normalized_x = preprocessing.normalize(x)
    print('Normalization complete')
    print()
    return normalized_x

def read_csv(columns_to_drop, column_label_name, csv_file_name):
    print('Reading csv file...')
    data = pd.read_csv('data/' + csv_file_name)
    data = data[~data.isin([np.nan, np.inf, -np.inf, np.negative]).any(1)]
    x = data.drop(columns_to_drop, axis=1)
    y = data[column_label_name]
    print('Reading csv file complete...')
    print('Dataset contains ', x.shape[1], 'features, and ', y.shape[0], 'registers')
    print()
    return x,y

def train_using_model(training_data, training_target, test_data, test_target, model_informed):
    model = train_model(training_data, training_target, model_informed)
    results = predict_values(test_data, model)
    visualize_predictions(results)
    print("Confusion Matrix: ")
    print(confusion_matrix(test_target, results))
    print("F-Score: ")
    print(f1_score(test_target, results, average='macro'))
    print("Precision: ")
    print(precision_score(test_target, results, average='macro'))
    print("Recall: ")
    print(recall_score(test_target, results, average='macro'))
    print("Accuraccy: ")
    print(accuracy_score(test_target, results))

def help():
    print('Supported arguments:')
    print('rf    - Random Forest')
    print('boost - Gradient Boosting')
    print('nb    - Gaussian NB')
    print('knn   - K Neighbors')
    print('dt    - Decision Tree')
    sys.exit()

def feature_selection(x_train, y_train, k):
    x_k_best= SelectKBest(f_classif, k=k).fit(x_train, y_train)

    mask = x_k_best.get_support()
    good_features = []
    bad_features = []
    for bool, feature in zip(mask, x_train.columns):
        if bool:
            good_features.append(feature)
        else:
            bad_features.append(feature)

    print('\nThe best features are:{}\n'.format(good_features))
    return good_features,bad_features