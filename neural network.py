# %%
# Cell 1: Load the numpy libraries
import numpy as np
import os
import keras
import csv
from keras.layers import BatchNormalization, Dense, Input, LeakyReLU
from keras.models import Model
# Import EarlyStopping callback from Keras
from keras.callbacks import EarlyStopping
import time
# Cell 2: Load data files
from datetime import datetime
from tensorflow.python.platform import build_info as tf_build_info
import tensorflow as tf
import tensorflow.keras.backend as K

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


def adjacent_accuracy(y_true, y_pred):
    """
    Custom accuracy metric that accepts predictions within ±1 of the true class.
    Args:
        y_true: True labels (integer values).
        y_pred: Predicted probabilities for each class (softmax output).
    Returns:
        Accuracy metric allowing adjacent values.
    """
    # Get the predicted class (argmax of softmax output)
    y_pred_classes = K.argmax(y_pred, axis=-1)

    # Ensure both tensors are of the same type
    y_true = K.cast(y_true, dtype='float32')  # Cast y_true to float32
    y_pred_classes = K.cast(y_pred_classes, dtype='float32')  # Cast y_pred_classes to float32 

    # Check if predictions are within ±1 of the true labels
    correct = K.abs(y_true - y_pred_classes) <= 1

    # Calculate mean accuracy
    return K.mean(K.cast(correct, dtype='float32'))



# Generate a random seed based on the current date
current_date = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
seed = int(current_date)  # Convert to integer

# Set random seed for reproducibility
#seed_gen = keras.random.SeedGenerator(seed=seed)

# Subdirectory with the module data
path = "C:/Users/darre/OneDrive/Documents/Work/Project/13_09_2021_ColdTests/"
path2 = "C:/Users/darre/OneDrive/Documents/Work/Project/181021_ColdTest/"
csvpath = "C:/Users/darre/OneDrive/Documents/Work/Project/"

datasets1 = {
    "VP0": {
        "VP0-0": {
            "trim0Mean": path + "Module0_VP0-0_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP0-0_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP0-0_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP0-0_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP0-0_Matrix_Mask.csv",
            "trim": path + "Module0_VP0-0_Matrix_Trim.csv",
        },
        "VP0-1": {
            "trim0Mean": path + "Module0_VP0-1_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP0-1_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP0-1_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP0-1_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP0-1_Matrix_Mask.csv",
            "trim": path + "Module0_VP0-1_Matrix_Trim.csv",
        },
        "VP0-2": {
            "trim0Mean": path + "Module0_VP0-2_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP0-2_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP0-2_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP0-2_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP0-2_Matrix_Mask.csv",
            "trim": path + "Module0_VP0-2_Matrix_Trim.csv",
        },
    },
    "VP1": {
        "VP1-0": {
            "trim0Mean": path + "Module0_VP1-0_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP1-0_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP1-0_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP1-0_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP1-0_Matrix_Mask.csv",
            "trim": path + "Module0_VP1-0_Matrix_Trim.csv",
        },
        "VP1-1": {
            "trim0Mean": path + "Module0_VP1-1_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP1-1_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP1-1_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP1-1_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP1-1_Matrix_Mask.csv",
            "trim": path + "Module0_VP1-1_Matrix_Trim.csv",
        },
        "VP1-2": {
            "trim0Mean": path + "Module0_VP1-2_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP1-2_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP1-2_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP1-2_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP1-2_Matrix_Mask.csv",
            "trim": path + "Module0_VP1-2_Matrix_Trim.csv",
        },
    },
    "VP2": {
        "VP2-0": {
            "trim0Mean": path + "Module0_VP2-0_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP2-0_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP2-0_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP2-0_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP2-0_Matrix_Mask.csv",
            "trim": path + "Module0_VP2-0_Matrix_Trim.csv",
        },
        "VP2-1": {
            "trim0Mean": path + "Module0_VP2-1_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP2-1_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP2-1_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP2-1_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP2-1_Matrix_Mask.csv",
            "trim": path + "Module0_VP2-1_Matrix_Trim.csv",
        },
        "VP2-2": {
            "trim0Mean": path + "Module0_VP2-2_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP2-2_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP2-2_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP2-2_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP2-2_Matrix_Mask.csv",
            "trim": path + "Module0_VP2-2_Matrix_Trim.csv",
        },
    },
    "VP3": {
        "VP3-0": {
            "trim0Mean": path + "Module0_VP3-0_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP3-0_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP3-0_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP3-0_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP3-0_Matrix_Mask.csv",
            "trim": path + "Module0_VP3-0_Matrix_Trim.csv",
        },
        "VP3-1": {
            "trim0Mean": path + "Module0_VP3-1_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP3-1_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP3-1_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP3-1_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP3-1_Matrix_Mask.csv",
            "trim": path + "Module0_VP3-1_Matrix_Trim.csv",
        },
        "VP3-2": {
            "trim0Mean": path + "Module0_VP3-2_Trim0_Noise_Mean.csv",
            "trimFMean": path + "Module0_VP3-2_TrimF_Noise_Mean.csv",
            "trim0Width": path + "Module0_VP3-2_Trim0_Noise_Width.csv",
            "trimFWidth": path + "Module0_VP3-2_TrimF_Noise_Width.csv",
            "mask": path + "Module0_VP3-2_Matrix_Mask.csv",
            "trim": path + "Module0_VP3-2_Matrix_Trim.csv",
        },
    },
}

datasets2 = {
    "VP0": {
        "VP0-0": {
            "trim0Mean": path2 + "Module0_VP0-0_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP0-0_TrimF_Noise_Mean.csv",
            "trim0Width": path2+ "Module0_VP0-0_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP0-0_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP0-0_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP0-0_Matrix_Trim.csv",
        },
        "VP0-1": {
            "trim0Mean": path2 + "Module0_VP0-1_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP0-1_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP0-1_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP0-1_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP0-1_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP0-1_Matrix_Trim.csv",
        },
        "VP0-2": {
            "trim0Mean": path2 + "Module0_VP0-2_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP0-2_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP0-2_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP0-2_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP0-2_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP0-2_Matrix_Trim.csv",
        },
    },
    "VP1": {
        "VP1-0": {
            "trim0Mean": path2 + "Module0_VP1-0_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP1-0_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP1-0_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP1-0_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP1-0_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP1-0_Matrix_Trim.csv",
        },
        "VP1-1": {
            "trim0Mean": path2 + "Module0_VP1-1_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP1-1_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP1-1_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP1-1_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP1-1_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP1-1_Matrix_Trim.csv",
        },
        "VP1-2": {
            "trim0Mean": path2 + "Module0_VP1-2_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP1-2_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP1-2_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP1-2_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP1-2_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP1-2_Matrix_Trim.csv",
        },
    },
    "VP2": {
        "VP2-0": {
            "trim0Mean": path2 + "Module0_VP2-0_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP2-0_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP2-0_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP2-0_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP2-0_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP2-0_Matrix_Trim.csv",
        },
        "VP2-1": {
            "trim0Mean": path2 + "Module0_VP2-1_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP2-1_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP2-1_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP2-1_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP2-1_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP2-1_Matrix_Trim.csv",
        },
        "VP2-2": {
            "trim0Mean": path2 + "Module0_VP2-2_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP2-2_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP2-2_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP2-2_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP2-2_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP2-2_Matrix_Trim.csv",
        },
    },
    "VP3": {
        "VP3-0": {
            "trim0Mean": path2 + "Module0_VP3-0_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP3-0_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP3-0_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP3-0_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP3-0_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP3-0_Matrix_Trim.csv",
        },
        "VP3-1": {
            "trim0Mean": path2 + "Module0_VP3-1_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP3-1_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP3-1_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP3-1_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP3-1_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP3-1_Matrix_Trim.csv",
        },
        "VP3-2": {
            "trim0Mean": path2 + "Module0_VP3-2_Trim0_Noise_Mean.csv",
            "trimFMean": path2 + "Module0_VP3-2_TrimF_Noise_Mean.csv",
            "trim0Width": path2 + "Module0_VP3-2_Trim0_Noise_Width.csv",
            "trimFWidth": path2 + "Module0_VP3-2_TrimF_Noise_Width.csv",
            "mask": path2 + "Module0_VP3-2_Matrix_Mask.csv",
            "trim": path2 + "Module0_VP3-2_Matrix_Trim.csv",
        },
    },
}




def load_submodule_data(files):
    """
    Load data for a single submodule.

    Args:
        files: Dictionary containing file paths for a submodule.

    Returns:
        x: Input features for the submodule.
        y: Labels for the submodule.
    """
    tMean0 = np.genfromtxt(files["trim0Mean"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tMeanF = np.genfromtxt(files["trimFMean"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tWidth0 = np.genfromtxt(files["trim0Width"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tWidthF = np.genfromtxt(files["trimFWidth"], delimiter=",").astype(np.float16).reshape(256 * 256)
    mask = np.genfromtxt(files["mask"], delimiter=",").astype(np.float16).reshape(256 * 256)
    trim = np.genfromtxt(files["trim"], delimiter=",").reshape(256 * 256).astype(np.int8)

    # Combine inputs
    x = np.column_stack([tMean0, tMeanF, tWidth0, tWidthF, mask])
    y = trim
    return x, y

def load_all_submodules(datasets, suffix=""):
    """
    Load all submodules into a dictionary for easy access.

    Args:
        datasets: Dictionary containing datasets organized by VP and submodules.
        suffix: String to append to each submodule key to avoid overwriting.

    Returns:
        all_submodules: Dictionary with keys as submodule names and values as data (x, y).
    """
    all_submodules = {}

    for vp, submodules in datasets.items():
        for submodule, files in submodules.items():
            key = f"{vp}-{submodule}{suffix}"  # Add suffix to differentiate
            print(f"Loading data for {key}...")
            x, y = load_submodule_data(files)
            all_submodules[key] = {"x": x, "y": y}

    return all_submodules


submodules1 = load_all_submodules(datasets1, suffix="_1")  # Suffix for datasets1
submodules2 = load_all_submodules(datasets2, suffix="_2")  # Suffix for datasets2

# Combine all submodules
all_submodules = {**submodules1, **submodules2}

def preprocess_data(x, y):
    """
    Filters rows with mask > 0 and NaN values, then normalizes tWidth0 and tMean0.
    
    Args:
        x: Input dataset (features).
        y: Labels corresponding to the input dataset.
    
    Returns:
        x: Processed input dataset.
        y: Processed labels.
    """
    # Step 1: Filter rows where mask (column 4) <= 0
    valid_indices = x[:, 4] <= 0  # Boolean mask for rows with mask <= 0
    x = x[valid_indices]
    y = y[valid_indices]

    # Step 2: Filter out rows with NaN values
    valid_indices = ~np.isnan(x).any(axis=1)  # Check for NaN in any column of x
    x = x[valid_indices]
    y = y[valid_indices]

    # Step 3: Normalize tWidth0 (column 2) and tMean0 (column 0)
    tWidth0_mean = np.mean(x[:, 2])  # Calculate mean of tWidth0
    tMean0_mean = np.mean(x[:, 0])   # Calculate mean of tMean0
    
    x[:, 2] -= tWidth0_mean          # Normalize tWidth0 column
    x[:, 0] -= tMean0_mean           # Normalize tMean0 column

    print(f"Processed data shape: {x.shape}, Labels shape: {y.shape}")
    print(f"tWidth0 mean used for normalization: {tWidth0_mean}")
    print(f"tMean0 mean used for normalization: {tMean0_mean}")

    return x, y


def preprocess_all_submodules(all_submodules):
    """
    Preprocess all submodules by filtering and normalizing data.

    Args:
        all_submodules: Dictionary of submodules with raw data.

    Returns:
        preprocessed_submodules: Dictionary of submodules with preprocessed data.
    """
    preprocessed_submodules = {}
    
    for submodule, data in all_submodules.items():
        print(f"Preprocessing data for {submodule}...")
        x, y = preprocess_data(data["x"], data["y"])  # Use your preprocess_data function
        preprocessed_submodules[submodule] = {"x": x, "y": y}

    return preprocessed_submodules

# Preprocess all data
all_submodules_preprocessed = preprocess_all_submodules(all_submodules)

# Separate odd and even submodules
odd_submodules = {key: value for i, (key, value) in enumerate(all_submodules_preprocessed.items()) if i % 2 == 0}
even_submodules = {key: value for i, (key, value) in enumerate(all_submodules_preprocessed.items()) if i % 2 != 0}

# Stack training data from odd submodules
x_train = np.vstack([data["x"] for data in even_submodules.values()])
y_train = np.hstack([data["y"] for data in even_submodules.values()])

# Stack evaluation data from even submodules
x_eval = np.vstack([data["x"] for data in odd_submodules.values()])
y_eval = np.hstack([data["y"] for data in odd_submodules.values()])

batch_size = 2 * 256
patience = 10
num_repeats = 3  # Number of times to repeat training for the same configuration

# Write results to a CSV file
csv_file = csvpath + "model_data_halfsplit.csv"

# Define a single layer combination for testing multiple times
layer1, layer2 = 'relu', 'tanh' 


# Prepare results storage
results = []




# Loop to repeat the configuration multiple times
for i in range(1):
    # Define the model
    nVal = x_train.shape[1]
    inputs = Input(shape=(nVal,))
    neurons = 2 * nVal
    norm = BatchNormalization()(inputs)
    # First hidden layer
    layer = Dense(neurons, activation=layer1.lower())(norm)
    # Second hidden layer
    layer = Dense(neurons, activation=layer2.lower())(layer)
    # Output layer
    outputs = Dense(16, activation='softmax')(layer)
    #outputs = Dense(1, activation = 'sigmoid')(layer)
    model = Model(inputs=inputs, outputs=outputs)
    # Compile the model
    model.compile(optimizer=keras.optimizers.get("Adam"),  # Adam optimizer
                  loss = keras.losses.SparseCategoricalCrossentropy(),  # Loss function: Sparse categorical crossentropy
                  metrics=[adjacent_accuracy]
                  )  # Metrics to track: accuracy
    # Define early stopping
    early_stopping = EarlyStopping(monitor="loss", patience=patience)
    # Measure training time
    start_time = time.time()
    history = model.fit(x=x_train, y=y_train,
                        batch_size=batch_size,
                        verbose=1,  # Output progress info
                        epochs=1000,  # Set max number of epochs, will stop earlier if needed
                        shuffle=True,  # Shuffle the training data before each epoch
                        callbacks=[early_stopping])  # Use early stopping to stop training if needed
    training_time = time.time() - start_time
    # Evaluate the model
    score = model.evaluate(x_eval, y_eval, verbose=1)
    final_accuracy = score[1]
    final_loss = score[0]
    total_closeness =  final_loss - final_accuracy  # Example metric
    # Log the results (including seed)
    results.append({
        'Iteration': i,
        'Input Layers': 2,
        'Layer Type': f"{layer1} & {layer2}",
        'Batch Size': batch_size,
        'Patience': patience,
        'Neurons': neurons,
        'Final Accuracy': final_accuracy,
        'Final Loss': final_loss,
        'Total Closeness': total_closeness,
        'Training Time (s)': training_time,
        'Training data': ', '.join(even_submodules.keys()),
        'Evaluation data': ', '.join(odd_submodules.keys()),
    })

# Check if CSV exists
file_exists = os.path.isfile(csv_file)
# Write all results to the CSV file
with open(csv_file, mode='a', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    if not file_exists:  # Write header only if the file is new
        writer.writeheader()
    writer.writerows(results)  # Write all rows at once
print(f"All results appended to {csv_file}")

