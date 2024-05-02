import csv

def save_results(history, test_loss, test_accuracy):
    csv_file = "results.csv"
    with open(csv_file, mode="w", newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Train Loss", "Train Accuracy", "Validation Loss", "Validation Accuracy", "Test Loss", "Test Accuracy"])
        writer.writerow([history.history['loss'],
                         history.history['accuracy'],
                         history.history["val_loss"], 
                         history.history["val_accuracy"], 
                         test_loss, test_accuracy])