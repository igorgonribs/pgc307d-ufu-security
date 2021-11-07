import sys
from termcolor import colored

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import utils as util

label_column_name = 'Label'
dropped_features = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 'Destination Port', 'Timestamp', 'Init_Win_bytes_backward', 'Flow Bytes/s', 'Label']
k = 10

rate_training = 0.7
test_training = 0.3

default_list_algoritms = {
        'rf': {'name': 'Random Forest', 'model':RandomForestClassifier()}, 
        'boost': {'name': 'Gradient Boosting', 'model':GradientBoostingClassifier()}, 
        'nb': {'name': 'Gaussian NB', 'model':GaussianNB()}, 
        'knn': {'name': 'K Neighbors', 'model':KNeighborsClassifier()}, 
        'dt': {'name': 'Decision Tree', 'model':DecisionTreeClassifier()}
    }

argument_list = sys.argv[1:]

if len(argument_list) == 0:
    argument_list = list(default_list_algoritms. keys())
    print('Executing the following training algoritms: ', str(argument_list))

if argument_list[0] == 'help':
    util.help()

#csv_file_name = '_smaller_dataset.csv'
csv_file_name = '_Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

x, y = util.read_csv(dropped_features, label_column_name, csv_file_name)
features_selected, features_to_drop = util.feature_selection(x, y, k)
normalized_x = util.normalize_features(x.drop(features_to_drop, axis=1))

training_data, training_target, test_data, test_target = util.set_trainingset_and_testset(y, normalized_x, rate_training, test_training)

for val in argument_list:
    print(colored(default_list_algoritms.get(val).get('name'), 'green'))
    util.train_using_model(training_data, training_target, test_data, test_target, default_list_algoritms.get(val).get('model'))