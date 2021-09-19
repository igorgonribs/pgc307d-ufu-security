import operation_functions as of

### Main function
rate_training = 0.9
test_training = 0.1

# columns_considered = [2, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
# columns excluded: 0, 1, 3, 6, 20, 21
columns_considered = [22, 23, 24, 25]
column_label_index = [84]

csv_file_name = '_smaller_dataset.csv'
#csv_file_name = '_Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'

x, y = of.read_csv(columns_considered, column_label_index, csv_file_name)
normalized_x = of.normalize_features(x)
training_data, training_target, test_data, test_target = of.set_trainingset_and_testset(y, normalized_x, rate_training, test_training)
model = of.train_model_random_forest(training_data, training_target)
results = of.predict_values(test_data, model)
of.calculate_hit_rate(test_target, results)
of.visualize_predictions(results)