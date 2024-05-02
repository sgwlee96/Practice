import csv

def save_results(history, test_loss, test_accuracy, csv_file, append=False):
    csv_file = "results.csv"
    with open(csv_file, mode="a" if append else 'w', newline='') as file:
        writer = csv.writer(file)

        if not append:

            writer.writerow(["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"])

        writer.writerow([history.history['loss'][-1],
                         history.history['accuracy'][-1],
                         history.history["val_loss"][-1], 
                         history.history["val_accuracy"][-1], 
                         test_loss, test_accuracy])