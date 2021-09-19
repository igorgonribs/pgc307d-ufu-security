import sys
from termcolor import colored

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import utils as util

# columns_considered = [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# columns excluded: 0, 1, 3, 6, 20, 21
columns_considered = [22, 23, 24, 25]
column_label_index = [84]

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

x, y = util.read_csv(columns_considered, column_label_index, csv_file_name)
normalized_x = util.normalize_features(x)
training_data, training_target, test_data, test_target = util.set_trainingset_and_testset(y, normalized_x, rate_training, test_training)

for val in argument_list:
    print(colored(default_list_algoritms.get(val).get('name'), 'green'))
    util.train_using_model(training_data, training_target, test_data, test_target, default_list_algoritms.get(val).get('model'))