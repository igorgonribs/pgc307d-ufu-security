from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import pandas as pd

rate_training = 0.9
test_training = 0.1

# columns_considered = [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# columns excluded: 0, 1, 3, 6, 20, 21
columns_considered = [22, 23, 24, 25]
column_label_index = [84]

#csv_file_name = '_smaller_dataset.csv'
csv_file_name = '_Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

print('Reading csv file...')
data = pd.read_csv('data/' + csv_file_name)
X = data.iloc[:, columns_considered].values
Y = data.iloc[:, column_label_index].values
print('Reading csv file complete...')

print('Normalizing data...')
normalizedX = preprocessing.normalize(X)
print('Normalization complete')

training_size = rate_training*len(Y)
test_size = test_training*len(Y)
print('The training set contains %2d registers' % training_size)
print('The testing set contains %2d registers' % test_size)

print('Setting training and test data...')
training_data = normalizedX[:int(training_size)]
training_target = Y[:int(training_size)]
test_data = normalizedX[int(-test_size):]
test_target = Y[int(-test_size):]

print('Starting training...')
model = RandomForestClassifier()
model.fit(training_data, training_target)
print('Training complete...')

print('Starting prediction...')
results = model.predict(test_data)
print('Prediction complete...')

hits = (results == test_target[:,0])

hit_rate = 100.0 * sum(hits)/test_size
print("Our algorithm hit rate: %f" % hit_rate)

## Visualization of the predicted values
## DDoS predictions
ddos = tuple(filter(lambda x: x == 'DDoS', results))
print("Number of DDoS predictions: %2d" % len(ddos))

## Benign predictions
benign = tuple(filter(lambda x: x == 'BENIGN', results))
print("Number of Benign predictions: %2d" % len(benign))