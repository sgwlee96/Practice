import csv

def save_results(model_name, history, test_loss, test_accuracy, csv_file, append=False):
    csv_file = "results.csv"
    with open(csv_file, mode="a" if append else 'w', newline='') as file:
        writer = csv.writer(file)

        if not append:

            writer.writerow(["Model","Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"])

        writer.writerow([model_name,
                         history.history['loss'][-1],
                         history.history['accuracy'][-1],
                         history.history["val_loss"][-1], 
                         history.history["val_accuracy"][-1], 
                         test_loss, test_accuracy])