import time
import sys

from sklearn import preprocessing
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
    hits = (results == test_target.values[:,0])

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

def set_trainingset_and_testset(y, normalized_x, rate_training, test_training):
    print('Setting training and test data...')
    training_size = rate_training*len(y)
    test_size = test_training*len(y)
    print('The training set contains %2d registers' % training_size)
    print('The testing set contains %2d registers' % test_size)
    print()
    training_data = normalized_x[:int(training_size)]
    training_target = y[:int(training_size)]
    test_data = normalized_x[int(-test_size):]
    test_target = y[int(-test_size):]
    return training_data,training_target,test_data,test_target

def normalize_features(x):
    print('Normalizing data...')
    normalized_x = preprocessing.normalize(x)
    print('Normalization complete')
    print()
    return normalized_x

def read_csv(columns_considered, column_label_index, csv_file_name):
    print('Reading csv file...')
    data = pd.read_csv('data/' + csv_file_name)
    x = data.iloc[:, columns_considered]
    y = data.iloc[:, column_label_index]
    print('Reading csv file complete...')
    print('Dataset contains ', len(columns_considered), 'features, and ', len(y), 'registers')
    print()
    return x,y

def train_using_model(training_data, training_target, test_data, test_target, model_informed):
    model = train_model(training_data, training_target, model_informed)
    results = predict_values(test_data, model)
    evaluate_model(test_target, results)

def evaluate_model(test_target, results):
    calculate_hit_rate(test_target, results)
    visualize_predictions(results)
    print()

def help():
    print('Supported arguments:')
    print('rf    - Random Forest')
    print('boost - Gradient Boosting')
    print('nb    - Gaussian NB')
    print('knn   - K Neighbors')
    print('dt    - Decision Tree')
    sys.exit()