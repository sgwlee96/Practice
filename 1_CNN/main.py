import model
import utils
import os.path

def main():
    csv_file = "results.csv"
    file_exists = os.path.isfile(csv_file)

    train_img, train_label, validate_img, validate_label, test_img, test_label = model.load_data()
    model_instance = model.build_model()
    history = model_instance.fit(train_img, train_label, epochs=10, batch_size=16, validation_data=(validate_img, validate_label))
    test_loss, test_accuracy = model_instance.evaluate(test_img, test_label)

    utils.save_results(history, test_loss, test_accuracy, csv_file, append=file_exists)


if __name__ == "__main__":
    main()