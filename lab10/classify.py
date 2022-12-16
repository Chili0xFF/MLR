import os
# Disable TF messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2     # pip install opencv-python
import argparse

# Default trained model file name (without extension)
MODEL_FILENAME = 'model'

# Construct the argument parser and parse the input arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=False,
                help="trained model file name (without the .h5 extension)",
                default=MODEL_FILENAME, metavar="filename")
ap.add_argument("-c", "--console", required=False, action='store_true', 
                help="display results in console only", default=False)
                
args = vars(ap.parse_args())
console_only = args["console"]
model_filename = args["model"]

# Load trained model of neural network
model = tf.keras.models.load_model(model_filename+".h5")

# Load and pre-process test set
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype('float32')/ 255
y_test_categorical = tf.keras.utils.to_categorical(y_test, 10)

# Evaluate the model using the whole test set
test_score = model.evaluate(x_test, y_test_categorical, verbose=0)
print("Accuracy: {:.2f}%, Precision: {:.2f}%, Recall: {:.2f}%".format(
    test_score[1]*100, test_score[2]*100, test_score[3]*100))
#F1 = ....

# Iterate over single test images
for i in range(len(x_test)):

    # Get the next image and its label from the test set
    image = x_test[i]
    actual_class = y_test[i]

    # Pre-process the loaded data
    image_data = np.expand_dims(image, axis=0)

    # Classify the input image
    prediction = model.predict(image_data, verbose=0)

    # Find the winner class and the probability
    winner_class = np.argmax(prediction)
    winner_probability = np.max(prediction)*100

    # Print results        
    print("Actual: {}, prediction: {} ({:.2f}%) \t{}".format(actual_class, winner_class, 
        winner_probability, "OK" if actual_class == winner_class else "Error!"))

    if not console_only:
        # Build the text labels
        label1 = f"Actual: {actual_class}"
        label2 = f"Predicted: {winner_class} ({winner_probability:.2f}%)"

        # Draw the labels on the image
        output_image = cv2.resize(image, (500,500))
        cv2.putText(output_image, label1, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        cv2.putText(output_image, label2, (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

        # Show the output image
        cv2.imshow("Output", output_image)
        if cv2.waitKey(0) & 0xFF == ord('q'): # Break on 'q' pressed
            break


# Zadanie 10 (4p)
#
# Uzupełnić program o obliczanie miary F1 na zbiorze testowym.
# 
# Znaleźć (metodą prób i błędów) dobry zestaw hiperparametrów (dający niską wartość F1):
# - architektura sieci (FC, LeNet5) 
# - przeprowadzanie augmentacji bądź nie
# - liczba epok (np. 5, 20, 50 100)
# Jako RANDOM_SEED przyjąć swój numer indeksu.
#
# Zliczać (programistycznie) błędne klasyfikacje (np. "spodziewano 4, otrzymano 9")
# Zapisać te błędne klasyfikacje posortowane od najczęściej występujących.
