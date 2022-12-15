import os
# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import cv2     # python -m pip install opencv-python
from keras.datasets import mnist
# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model filename
MODEL = 'model.h5'

if __name__ == "__main__":
    
    # Load trained model
    model = tf.keras.models.load_model(MODEL)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    #for image_name in os.listdir(TEST_DATASET_DIR):
    for image in x_test:  
        # Load the image
        #image = cv2.imread(TEST_DATASET_DIR + os.path.sep + image_name, cv2.IMREAD_GRAYSCALE)
        # Pre-process the image for classification
        image_data = image.astype('float32') / 255
        image_data = tf.keras.preprocessing.image.img_to_array(image_data)
        # Expand dimension (28,28,1) -> (1,28,28,1)
        image_data = np.expand_dims(image_data, axis=0)
        
        # Classify the input image
        prediction = model.predict(image_data)
        
        # Soting the array
        argsorted = np.argsort(prediction)
        # Find the winner class and the probability
        winner_class = argsorted[0][-1]
        winner_probability = prediction[0][winner_class]*100
        
        secondPlace = argsorted[0][-2]
        secondProp = prediction[0][secondPlace]*100
        
        trdPlace = argsorted[0][-3]
        trdProp = prediction[0][trdPlace]*100
        
        # Build the text label
        label = f"prediction = {winner_class} ({winner_probability:.2f}%)"
        
        # Draw the label on the image
        output_image = cv2.resize(image, (500,500))
        cv2.putText(output_image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        if secondProp > 1:
            label2 = f"prediction = {secondPlace} ({secondProp:.2f}%)"
            cv2.putText(output_image, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        if trdProp > 1:
            label3 = f"prediction = {trdPlace} ({trdProp:.2f}%)"
            cv2.putText(output_image, label3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
        # Show the output image        
        cv2.imshow("Output", output_image)
        
        # Break on 'q' pressed, continue on the other key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break


# Zadanie 9.2 (1p)
# Uzupełnić wyświetlany tekst na obrazie o klasy z miejsca 2 i 3, 
# o ile ich prawdopodobieństwo jest większe od 1%.

# Zadanie 9.3 (1.5p)
# Zamiast wczytywać obrazy testowe z plików, ładować je metodą mnist.load_data() z API Keras.

# Wynik: plik tekstowy z uzupełnionym kodem oraz plik graficzny z przykładowym wynikiem predykcji.