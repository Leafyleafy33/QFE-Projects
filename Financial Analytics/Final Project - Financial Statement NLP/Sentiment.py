
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
import re
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import time

class SentimentAnalysisModel:
    def __init__(self, dataframe, text_column):
        self.dataframe = dataframe
        self.text_column = text_column
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(2, 5), max_features=3000, min_df=1, max_df=0.4) # did a lot of testing to get these parameters
        self.item7_pattern = re.compile(
            r'item\s+7[.,]?\s+management\'?s?\s+discussion\s*(and|&)\s+analysis(\'?s)?\s*(of|on)?\s*financial\s+conditions?\s*(and|&)?\s*results\s*(of|on)?\s*operations?', 
            re.IGNORECASE
        )

    def preprocess(self):
        self.dataframe['cleaned_text'] = self.dataframe[self.text_column].str.lower().replace(self.item7_pattern, '', regex=True)

    def vectorize_text(self, X_train, X_test):
        return self.vectorizer.fit_transform(X_train), self.vectorizer.transform(X_test)


    def build_ffnn_model(self, input_dim):
        model = Sequential([
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01), input_dim=input_dim),
            Dropout(0.5),
            Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer=SGD(), loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train_model(self, X_train_vec, y_train, X_test_vec, y_test, model_type='nb'):
        if model_type == 'nb':
            model = MultinomialNB()
            model.fit(X_train_vec, y_train)
        else:
            model = self.build_ffnn_model(X_train_vec.shape[1])
            early_stopping = EarlyStopping(monitor='val_loss', patience=3000, min_delta=0.01, restore_best_weights=True)
            X_train_vec_dense = X_train_vec.toarray()
            history = model.fit(X_train_vec_dense, y_train, epochs=3300, batch_size=64, verbose=0, validation_split=0.3, callbacks=[early_stopping])
    
            self.plot_learning_curves(history)
    
            y_pred = (model.predict(X_test_vec.toarray()) > 0.5).astype("int32")
            
        y_pred = model.predict(X_test_vec.toarray()) if model_type != 'nb' else model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred.round())
        conf_matrix = confusion_matrix(y_test, y_pred.round())
        print(f"Accuracy for the {model_type.upper()} model: {accuracy:.2%}")
        print(f"F1 Score for the {model_type.upper()} model: {f1_score(y_test, y_pred.round()):.2%}")
        print(f"Precision for the {model_type.upper()} model: {precision_score(y_test, y_pred.round()):.2%}")
        print(f"Recall for the {model_type.upper()} model: {recall_score(y_test, y_pred.round()):.2%}")
        self.plot_confusion_matrix(conf_matrix, f"{model_type.upper()} Model")
        return accuracy, conf_matrix
    
    def train_lstm(self, X_train, y_train, X_test, y_test):
        myBatch = 256
        myEpochs = 100  
        max_tokens = 3000  
        embed_dimension = 50  
        lstm_units = 64  

        # Create the text vectorization layer
        vectorize_layer = TextVectorization(
            max_tokens=max_tokens,
            output_mode='int',
            ngrams=(2, 5),
            output_sequence_length=500)  

        # Apply the TextVectorization layer to the training texts
        vectorize_layer.adapt(X_train.to_numpy())

        # Define LSTM model architecture
        model = Sequential([
            vectorize_layer,
            Embedding(input_dim=max_tokens + 1, output_dim=embed_dimension, mask_zero=True),
            Bidirectional(LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Bidirectional(LSTM(units=32, return_sequences=True, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Bidirectional(LSTM(units=16, return_sequences=True, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Bidirectional(LSTM(units=8, return_sequences=True, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Bidirectional(LSTM(units=4, return_sequences=False, kernel_regularizer=l2(0.001))),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=SGD(), loss="binary_crossentropy", metrics=["accuracy"])

        lr_scheduler = ReduceLROnPlateau(factor=0.5, patience=3)

        # Fit the model
        history = model.fit(
            x=X_train,  
            y=y_train,
            epochs=myEpochs,
            batch_size=myBatch,
            verbose=0,
            validation_split=0.3,
            callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True), lr_scheduler]
        )
     

        # Plot learning curves
        self.plot_learning_curves(history)

        # Evaluate the model on the test data
        y_pred = (model.predict(X_test) > 0.5).astype("int32")

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        print(f"Accuracy for the LSTM model: {accuracy:.2%}")
        print(f"F1 Score for the LSTM model: {f1:.2%}")
        print(f"Precision for the LSTM model: {precision:.2%}")
        print(f"Recall for the LSTM model: {recall:.2%}")

        # Plot the confusion matrix
        self.plot_confusion_matrix(conf_matrix, "LSTM")

        return accuracy, f1, precision, recall, conf_matrix

        
    def plot_learning_curves(self, history):
        # Update the plotting parameters for a night mode theme
        night_theme_style = {
            'axes.facecolor': '#0a1029',  
            'axes.edgecolor': '#0a1029',  
            'axes.labelcolor': '#eccfac',
            'figure.facecolor': '#0a1029',  
            'text.color': '#eccfac',
            'xtick.color': '#eccfac',
            'ytick.color': '#eccfac',
            'axes.spines.right': False,
            'axes.spines.bottom': True,
            'axes.spines.left': True
        }
        plt.rcParams.update(night_theme_style)

        # Create the subplots
        fig, axs = plt.subplots(1, 2, figsize=(8.3, 6))

        # Plot accuracy over epochs with the updated color palette
        axs[0].plot(history.history['accuracy'], label='Train Accuracy', color="#1b998b")  
        axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy', color="#f4d35e")  
        axs[0].set_title('Accuracy over epochs', color='#eccfac')
        axs[0].set_xlabel('Epochs', color='#eccfac')
        axs[0].set_ylabel('Accuracy', color='#eccfac')
        axs[0].legend()

        # Plot loss over epochs with the updated color palette
        axs[1].plot(history.history['loss'], label='Train Loss', color="#ee964b")  
        axs[1].plot(history.history['val_loss'], label='Validation Loss', color="#115173")  
        axs[1].set_title('Loss over epochs', color='#eccfac')
        axs[1].set_xlabel('Epochs', color='#eccfac')
        axs[1].set_ylabel('Loss', color='#eccfac')
        axs[1].legend()
        axs[0].grid(False)
        axs[1].grid(False)
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix(self, cm, model_name):
        # Update the plotting parameters for a night mode theme with a custom color palette
        night_theme_style = {
            'axes.facecolor': '#0a1029',  
            'axes.edgecolor': '#0a1029',  
            'axes.labelcolor': '#eccfac',
            'figure.facecolor': '#0a1029',  
            'text.color': '#eccfac',
            'xtick.color': '#eccfac',
            'ytick.color': '#eccfac',
            'axes.spines.right': False,
            'axes.spines.bottom': True,
            'axes.spines.left': True
        }
        plt.rcParams.update(night_theme_style)

        # Inverting the color palette for the confusion matrix
        cmap = sns.diverging_palette(12, 250, s=75, l=40, n=9, center="dark", as_cmap=True)

        # Create the heatmap with the inverted color palette
        plt.figure(figsize=(10.2, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, cbar=True,
                    cbar_kws={"shrink": 0.45, "ticks": [0, cm.max()/2, cm.max()]}, 
                    linewidths=1, linecolor='#0a1029', 
                    annot_kws={"color": "#eccfac", "weight": "bold"})

        # Customize the plot to match the theme of the image
        plt.title(f'Confusion Matrix for {model_name}', fontsize=18, weight='bold', color='#eccfac')
        plt.xlabel('Predicted', fontsize=14, color='#eccfac')
        plt.ylabel('Actual', fontsize=14, color='#eccfac')

        # Show the plot
        plt.show()