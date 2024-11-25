# Load in packages 

# General packages
import pandas as pd
import numpy as np
import librosa
import os
import librosa.util
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize, LabelBinarizer
import time

# Hyper parameters 
import optuna

# Time series
from aeon.classification.convolution_based import MultiRocketHydraClassifier
from aeon.classification.hybrid import HIVECOTEV2

# CNN
import tensorflow as tf
from tensorflow.image import resize
from keras.applications import DenseNet169, VGG19, ResNet152V2
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, GlobalMaxPool2D
from tensorflow.image import resize
from keras.optimizers import Adam

# Memory measurment 
import tracemalloc



# Set up for pipeline
# CNN: LogMelSpec, MFCC, 
# TSC: Audio
feature = "LogMelSpec"
# Sampling rate
# CNN: 44100 
# TSC: 300
sr = 22050
# Target shape 
# CNN: 224 
# TSC: None
target_shape = 224
# Flattening the data
flatten = False
# Number of epochs
epochs = 20 


# Paths
cd = os.getcwd()
# Meta data
tracks = os.path.join(cd, "Data/fma_small_meta/small_tracks.csv")
# Path to datasets 
songs = os.path.join(cd, "Data/fma_small")

# Loading meta data
training_set = pd.read_csv(os.path.join(cd, "Data/fma_small_meta/small_tracks_train.csv"), index_col=0, header=[0,1])[:1000]
validation_set = pd.read_csv(os.path.join(cd, "Data/fma_small_meta/small_tracks_val.csv"), index_col=0, header=[0,1])[:200]
testing_set = pd.read_csv(os.path.join(cd, "Data/fma_small_meta/small_tracks_test.csv"), index_col=0, header=[0,1])[:200]


# Overview of functions used in pre-processing of the data

# Proper formatting of track_id
def adding_zeros(track_id):
    # Padding the track id with zeros to match the MP3 file naming convention (6 digits)
    formatted_track_id = f"{track_id:06d}"
    return formatted_track_id

def get_input_path(track_id, folder):
    # Formatting the track id to match the naming of the MP3 files
    track_id = adding_zeros(track_id)

    # Input path for the song
    audio_path = os.path.join(songs, folder, f"{track_id}.mp3")

    return audio_path

def get_audio_padded_clipped(audio_path, sec_len=30, sr=22050):
    # Load the audio file into a NumPy array with the default sampling rate (22050 Hz)
    audio, sr = librosa.load(audio_path, sr=sr)

    # 1 second of audio if sampling rate is 22050 Hz this is 30 sec
    desired_length = sr * sec_len 

    if len(audio) < desired_length:
        # Pad with zeros if shorter
        audio = np.pad(audio, (0, desired_length - len(audio)), 'constant')
    else:
        # Clipping if longer
        audio = audio[:desired_length]
    return audio, sr

def get_log_MelSpec(audio, sr=22050, flatten=False):
    # Compute the Mel-spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    # Convert the Mel-spectrogram to log scale
    log_mel = librosa.amplitude_to_db(mel)

    # Make data one-dimensional if needed
    if flatten:
        return log_mel.flatten()
    else:
        return log_mel
    
def get_MFCC(audio, sr=22050, flatten=False):
    # Compute the Mel-frequency cepstral coefficients (MFCC)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

    # Make data one-dimensional if needed
    if flatten:
        return mfcc.flatten()
    else:
        return mfcc

def reshape(feature, target_shape, channels=3):
    # Adding channel dimension 
    expanded_feature = np.stack([feature ]*channels, axis=-1)
    # Reshaping to target size
    resized_feature = resize(expanded_feature, (target_shape, target_shape))
    return resized_feature



# Audio processsor class to combine all the pre-processing steps
class AudioProcessor:
    def __init__(self, dataset, feature, target_shape=None, flatten=False, setname=None, sr=44100):
        self.dataset = dataset
        self.feature = feature
        self.target_shape = target_shape
        self.flatten = flatten
        self.setname = setname
        self.sr = sr
        self.track = 0

    def __iter__(self):
        return self

    def __next__(self):
        # Stop iteration if there are no more tracks
        if self.track >= len(self.dataset):
            raise StopIteration
        
        # Get track_id from the dataset index
        track_id = self.dataset.index[self.track]
        
        # Construct the audio path for each track
        audio_path = get_input_path(track_id, self.setname)

        # Get audio data with the current sample rate
        audio_features, _ = get_audio_padded_clipped(audio_path, sr=self.sr)

        # Apply feature extraction based on the given feature type
        if self.feature == "MFCC":
            feature = get_MFCC(audio_features, flatten=self.flatten)
        elif self.feature == "LogMelSpec":
            feature = get_log_MelSpec(audio_features, flatten=self.flatten)
        elif self.feature == "Audio":
            feature = audio_features
        else:
            raise ValueError(f"Unsupported feature type: {self.feature}")

        # Reshape the feature if a target shape is provided
        if self.target_shape is not None:
            feature = reshape(feature, self.target_shape)
        
        # Move to the next track
        self.track += 1

        return feature

def binarizer(y):
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    return y, len(lb.classes_), lb.classes_



# Hyper parameter tuning
def optimize_cnn(trial: optuna.trial):
    # Optimize dense layers
    dense_units1 = trial.suggest_categorical('dense_units1', [32, 64, 128, 256, 512, 1024])
    dense_units2 = trial.suggest_categorical('dense_units2', [32, 64, 128, 256, 512, 1024])
    dense_units3 = trial.suggest_categorical('dense_units3', [32, 64, 128, 256, 512, 1024])
    # Optimize dropout
    dropout_rate = trial.suggest_uniform('dropout', 0.0, 0.5)
    # Optimize learning rate
    lr = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    # Optimize batch size
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    # Optimize Epochs
    epochs = trial.suggest_int('epochs', 5, 20)

    # Creating model and adding trail variables
    model = create_model(input_shape=(224, 224, 3), num_classes=num_classes, dense_units1=dense_units1, dense_units2=dense_units2, dense_units3=dense_units3, dropout_rate=dropout_rate, lr=lr)

    # Fitting model
    model.fit(X_train, y_train_b, epochs = epochs, batch_size = batch_size, verbose = 0)

    # Evaluate model performance
    val_loss, val_acc = model.evaluate(X_val, y_val_b, verbose=0)

    # return the validation set accuracy to determine the optimization direction
    return val_acc 


def optimize_mr(trail: optuna.trial):
    # optimize number of kernels per group for the Hydra transform
    kernels = trail.suggest_categorical('n_kernels', [1, 2, 4, 8, 16, 32])
    # optimize number of groups per dilation for the Hydra transform
    groups = trail.suggest_categorical('n_groups', [1, 2, 4, 8, 16, 32])

    # Training and evaluating the classifier
    clf = MultiRocketHydraClassifier(n_kernels=kernels, n_groups=groups) 

    clf.fit(X_train, y_train)
    # Predict and evaluate accuracy on validation set
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    # Return the accuracy to be maximized by Optuna
    print(accuracy)
    return accuracy



# Function to create CNN model 
def create_model(input_shape=(224, 224, 3), num_classes=8, dense_units1=1024, dense_units2=512, dense_units3=256, dropout_rate=0.5, lr=0.001):
    # Load the transfer learning model without the top fully connected layers
    base_model = VGG19(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the layers of the base model
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom classification layers on top of the base
    x = base_model.output
    x = GlobalMaxPool2D()(x)  # Flatten the output of the convolutional base
    x = Dense(dense_units1, activation='relu')(x) 
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(dense_units2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)  # Add dropout for regularization
    x = Dense(dense_units3, activation='relu')(x)
    output = Dense(num_classes, activation='softmax')(x) 
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=output)
    model.compile(loss='categorical_crossentropy', 
                  optimizer=Adam(learning_rate=lr), 
                  metrics=['accuracy'])

    return model



# Functions use for plotting results
def classification(y_pred, y_true, model_name, feature):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted') 
    recall = recall_score(y_true, y_pred, average='weighted') 
    f1 = f1_score(y_true, y_pred, average='weighted') 
    
    print(f"accuracy {feature}, {model_name}: {accuracy}")
    print(f"precision {feature}, {model_name}: {precision}")
    print(f"recall {feature}, {model_name}: {recall}")
    print(f"f1 {feature}, {model_name}: {f1}")

def plot_confusion_matrix(y_true, y_pred, model_name, classes, feature):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    
    # Rotate x-axis tick labels to vertical
    plt.xticks(rotation=90)

    plt.title(f'{model_name} {feature} Confusion Matrix')
    plt.savefig(f'{model_name}_{feature}_confusion_matrix.png')
    plt.show()

def plot_confusion_matrix_cnn(y_true, y_pred, model_name, classes, feature):
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    
    # Rotate x-axis tick labels to vertical
    plt.xticks(rotation=90)

    plt.title(f'{model_name} {feature} Confusion Matrix')
    plt.savefig(f'{model_name}_{feature}_confusion_matrix.png')
    plt.show()

def plot_history(history, model_name, feature):
    # Extract data from history object
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    accuracy = history.history.get('accuracy', [])
    val_accuracy = history.history.get('val_accuracy', [])

    epochs = range(1, len(loss) + 1)

    # Define colors
    training_color = '#1f77b4'  # Blue
    validation_color = '#4a90e2'  # Lighter blue

    # Create a figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 6), dpi=100)
    fig.suptitle('Training History', fontsize=20, fontweight='bold', color=training_color)

    # Plot Loss
    ax[0].plot(epochs, loss, marker='o', color=training_color, label='Training Loss', linewidth=2)
    if val_loss:
        ax[0].plot(epochs, val_loss, marker='o', color=validation_color, label='Validation Loss', linewidth=2)
    ax[0].set_title('Loss Over Epochs', fontsize=16, fontweight='bold', color=training_color)
    ax[0].set_xlabel('Epochs', fontsize=14)
    ax[0].set_ylabel('Loss', fontsize=14)
    ax[0].legend(fontsize=12)
    ax[0].grid(True, linestyle='--', alpha=0.6)

    # Plot Accuracy
    ax[1].plot(epochs, accuracy, marker='o', color=training_color, label='Training Accuracy', linewidth=2)
    if val_accuracy:
        ax[1].plot(epochs, val_accuracy, marker='o', color=validation_color, label='Validation Accuracy', linewidth=2)
    ax[1].set_title('Accuracy Over Epochs', fontsize=16, fontweight='bold', color=training_color)
    ax[1].set_xlabel('Epochs', fontsize=14)
    ax[1].set_ylabel('Accuracy', fontsize=14)
    ax[1].legend(fontsize=12)
    ax[1].grid(True, linestyle='--', alpha=0.6)

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the main title

    # Save the figure
    plt.savefig(f'{model_name}_{feature}_curve.png')

    # Show the plot
    plt.show()



# Experiment execution

# Start measurement of time and memory
tracemalloc.start()
feature_start = time.time()

# Creating each set X and y 
feature_train = AudioProcessor(dataset=training_set, 
                                       feature=feature, 
                                       target_shape=target_shape, 
                                       flatten=flatten, 
                                       setname="training_set", 
                                       sr=sr
)
feature_val = AudioProcessor(dataset=validation_set, 
                                       feature=feature, 
                                       target_shape=target_shape, 
                                       flatten=flatten, 
                                       setname="validation_set", 
                                       sr=sr
)
feature_test = AudioProcessor(dataset=testing_set, 
                                       feature=feature,  
                                       target_shape=target_shape, 
                                       flatten=flatten, 
                                       setname="testing_set", 
                                       sr=sr
)


# Collect all audio features into a list
X_train = [audio_features for audio_features in feature_train]
X_val = [audio_features for audio_features in feature_val]
X_test = [audio_features for audio_features in feature_test]

# Convert the list of features into a numpy array 
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# Getting all labels for y
y_train = training_set["track"]["genre_top"]
y_val = validation_set["track"]["genre_top"]
y_test = testing_set["track"]["genre_top"]

# timing
feature_end = time.time()

# split for either CNN or TSC set up
if (feature == "LogMelSpec") | (feature == "MFCC"):
    # Binarization of data
    y_train_b, num_classes, classes = binarizer(y_train)
    y_val_b, _, _ = binarizer(y_val)
    y_test_b, _, _ = binarizer(y_test)

    # Best parameter set up 
    model = create_model(input_shape=(224, 224, 3), num_classes=num_classes, dense_units1=256, dense_units2=64, dense_units3=32, dropout_rate=0.49005267786325046, lr=0.000497874098197339)

    # Training model and get validation/test results 
    start_time_train = time.time()
    model.fit(X_train, y_train_b, epochs = epochs, batch_size = 32, verbose = 0)
    end_time_train = time.time()
    # Get predictions as probabilities
    start_time_pred = time.time()
    y_pred_probs = model.predict(X_test)
    end_time_pred = time.time()
    # Convert to binary format
    y_pred = (y_pred_probs == np.max(y_pred_probs, axis=1, keepdims=True)).astype(int)


    print(f"create feature time {feature}, {feature_end-feature_start} seconds")
    print(f"train time first {feature}, first: {end_time_train-start_time_train} seconds")
    print(f"pred time first {feature}, first: {end_time_pred-start_time_pred} seconds")

    # Creating and saving plots
    classification(y_test_b, y_pred, "first", feature)
    plot_confusion_matrix_cnn(y_test_b, y_pred, "first", np.unique(y_train), feature)



else: 
    # Choose either model
    clf = MultiRocketHydraClassifier(n_kernels=32, n_groups=16)
    clf = HIVECOTEV2(time_limit_in_minutes=1440)#1440 = 24H

    # Training model and get validation/test results 
    start_time_train = time.time()
    clf.fit(X_train, y_train)
    end_time_train = time.time()
    # Predict and evaluate accuracy on validation/test set
    start_time_pred = time.time()
    y_pred = clf.predict(X_test)
    end_time_pred = time.time()

    print(f"create feature time {feature}, {feature_end-feature_start} seconds")
    print(f"train time:{end_time_train-start_time_train} seconds")
    print(f"pred time:{end_time_pred-start_time_pred} seconds")

    # Creating and saving plots
    classification(y_test, y_pred, "MR-H", feature)
    plot_confusion_matrix(y_test, y_pred, "MR-H", np.unique(y_train), feature)




current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.2f} MB; Peak was: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()
