import os
# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
EPOCHS = 5
MODEL_FILENAME = "model.h5"
NO_OF_CLASSES = 10
VAL_SPLIT = 0.2
HIDDEN_UNITS = 50

class FullyConnectedForMnist:
    '''Simple NN for MNIST database. INPUT => FC/RELU => FC/SOFTMAX'''
    def build(hidden_units):
        # Initialize the model
        model = tf.keras.models.Sequential()
        # Flatten the input data of (x, y, 1) dimension
        model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
        # FC/RELU layer
        model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        # Softmax classifier (10 classes)
        model.add(tf.keras.layers.Dense(NO_OF_CLASSES, activation="softmax"))
        return model


if __name__ == "__main__":
    
    # Load dataset as train and test sets
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Convert from uint8 to float32 and normalize to [0,1]
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # Transform labels to 'one-hot' encoding, e.g.
    # 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    y_train = tf.keras.utils.to_categorical(y_train, NO_OF_CLASSES)
    y_test = tf.keras.utils.to_categorical(y_test, NO_OF_CLASSES)

    # Construct the model
    model = FullyConnectedForMnist.build(HIDDEN_UNITS)

    # Compile the model and print summary
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    # Train the model
    archive = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_split=VAL_SPLIT)
    print(archive.history.keys())
    print(archive.history['loss'])
    print(archive.history['accuracy'])
    loss=archive.history['loss']
    accuracy=archive.history['accuracy']
    plt.plot(loss)
    plt.title('Result')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss.png')
    plt.show()
    plt.plot(accuracy)
    plt.title('Resul')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy.png')
    plt.show()
    # Save model to a file
    model.save(MODEL_FILENAME)
    
    # Evaluate the model on the test data
    model.evaluate(x_test, y_test)


# Zadanie 9.1 (1.5p)
# Uzupełnić plik o rysowanie wykresu funkcji kosztu (loss) oraz dokładności (accuracy) 
# w funkcji epoki.

# Wynik: plik tekstowy z uzupełnionym kodem oraz plik graficzny z przykładowym wykresem.