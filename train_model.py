import os
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dropout, Dense, TimeDistributed, GlobalAveragePooling2D, LSTM

from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import EfficientNetB1, EfficientNetB3

class Classifier:
    def __init__(self, input_shape=(229, 229, 3), learning_rate=0.0001, retrain=False, save_model=True, trained_model='./models/model_inception_6.keras', selected_model=3, is_sequential_training=False):
        """
        Initializes the Classifier with specified parameters.

        Args:
            input_shape (tuple): The shape of the input images.
            learning_rate (float): The learning rate for training the model.
            retrain (bool): If True, loads a pre-trained model for further training.
            save_model (bool): If True, saves the trained model after training.
            trained_model (str): Path to the pre-trained model file.
            selected_model (int): Model selection identifier (1 for EfficientNetB1, 3 for EfficientNetB3).
            is_sequential_training (bool): If True, uses sequential training with LSTM layers.

        """
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.retrain = retrain
        self.trained_model = trained_model
        self.model_path = './models/'
        self.save_model = save_model
        self.selected_model = selected_model
        self.is_sequential_training = is_sequential_training
        self.model_config = {
                            "learning_rate": 0.0001,
                            "kernel_regularizer_l2":0.0001
                            }

        if self.retrain:
            if not os.path.exists(self.trained_model):
                raise FileNotFoundError(f"Trained model file not found: {self.LANDMARKS_MODEL_PATH}")
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.model = self.build_model()

    def build_model(self):
        """    
        Builds the model architecture based on the selected configuration.

        Returns:
            model: A compiled Keras model.
        
        Notes:
            - If `self.retrain` is True, loads an existing trained model.
            - Otherwise, constructs a new model using EfficientNet with optional LSTM layers.
        """
        img_size = self.input_shape
        if self.is_sequential_training:
            img_size = self.input_shape[1:]

        if self.retrain:
            model = tf.keras.models.load_model(self.trained_model)
        else:
            if self.selected_model == 1:
                pretrained_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=img_size)
            elif self.selected_model == 3:
                pretrained_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=img_size)
            else:
                pretrained_model = EfficientNetB1(weights='imagenet', include_top=False, input_shape=img_size)

            pretrained_model.trainable = False

            if self.is_sequential_training:
                print('inside sequential learning: ')
                model = Sequential([
                            Input(shape=self.input_shape),
                            TimeDistributed(pretrained_model),
                            TimeDistributed(GlobalAveragePooling2D()),  
                            LSTM(128, return_sequences=True),  
                            Dropout(0.5),
                            TimeDistributed(Dense(64, activation='relu')),  
                            TimeDistributed(Dense(1, activation='sigmoid'))  
                        ])
            else:
                model = Sequential([
                            Input(shape=self.input_shape),
                            pretrained_model,
                            GlobalAveragePooling2D(),
                            Dropout(0.5),
                            Dense(256, activation='relu', kernel_regularizer=l2(self.model_config['kernel_regularizer_l2'])),
                            Dense(1, activation='sigmoid') 
                        ])

        model.compile(optimizer=Adam(learning_rate=self.model_config["learning_rate"]), 
                      loss='binary_crossentropy', 
                      metrics=['accuracy'])
        
        return model

    def train(self, training_data, validation_data, batch_size=32, epochs=3):
        """
        Trains the model using the given training and validation datasets.

        Args:
            training_data (tuple): Tuple containing training features and labels.
            validation_data (tuple): Tuple containing validation features and labels.
            batch_size (int): Number of samples per training batch.
            epochs (int): Number of training iterations over the entire dataset.

        Returns:
            history: The training history object containing loss and accuracy metrics.
        """
        es = EarlyStopping(monitor='val_loss', verbose=1, patience=3, restore_best_weights=True)
        # print('training data: ', *training_data)
        history = self.model.fit(
            *training_data,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=[es])

        if self.save_model:
            self.save()

        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluates the trained model on test data.

        Args:
            X_test (array-like): Feature data for evaluation.
            y_test (array-like): Ground truth labels for evaluation.

        Returns:
            tuple: A tuple containing test loss and test accuracy.
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        
        return test_loss, test_acc
    
    def save(self):
        """
        Saves the trained model to a specified directory.

        Returns:
            None
        """
        
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        
        model_file_path = os.path.join(self.model_path, f'model_efficient.keras')
        self.model.save(model_file_path)
        print(f"Model saved at: {model_file_path}")