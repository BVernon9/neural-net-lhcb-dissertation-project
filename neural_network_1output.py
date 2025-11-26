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
import tensorflow.keras.backend as K
import tensorflow as tf

'''def adjacent_accuracy(y_true, y_pred):
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
'''
    
def adjacent_accuracy(y_true, y_pred):
    """
    Custom accuracy metric for models with a single output neuron scaled to [0, 1].
    Allows predictions within ±1 of the true class.
    Args:
        y_true: True labels (scaled to [0, 1]).
        y_pred: Predicted values (continuous outputs scaled to [0, 1]).
    Returns:
        Accuracy metric allowing adjacent values.
    """
    # Step 1: Rescale predictions and true labels to [0, 15]
    y_pred_rescaled = K.round(y_pred * 15) 
    y_true_rescaled = K.round(y_true * 15)  

    # Step 2: Check if predictions are within ±1 of the true labels
    correct = K.abs(y_true_rescaled - y_pred_rescaled) <= 1

    # Step 3: Calculate mean accuracy
    return K.mean(K.cast(correct, dtype='float32'))



# Generate a random seed based on the current date
current_date = datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
seed = int(current_date)  # Convert to integer

# Set random seed for reproducibility
#seed_gen = keras.random.SeedGenerator(seed=seed)

# Subdirectory with the module data
path = "C:/Users/darre/OneDrive/Documents/Work/Project/13_09_2021_ColdTests/"
csvpath = "C:/Users/darre/OneDrive/Documents/Work/Project/"

# Define file paths for all VPs
datasets = {
    "VP0": {
        "trim0Mean": path + "Module0_VP0-0_Trim0_Noise_Mean.csv",
        "trimFMean": path + "Module0_VP0-0_TrimF_Noise_Mean.csv",
        "trim0Width": path + "Module0_VP0-0_Trim0_Noise_Width.csv",
        "trimFWidth": path + "Module0_VP0-0_TrimF_Noise_Width.csv",
        "mask": path + "Module0_VP0-0_Matrix_Mask.csv",
        "trim": path + "Module0_VP0-0_Matrix_Trim.csv",
    },
    "VP1": {
        "trim0Mean": path + "Module0_VP1-0_Trim0_Noise_Mean.csv",
        "trimFMean": path + "Module0_VP1-0_TrimF_Noise_Mean.csv",
        "trim0Width": path + "Module0_VP1-0_Trim0_Noise_Width.csv",
        "trimFWidth": path + "Module0_VP1-0_TrimF_Noise_Width.csv",
        "mask": path + "Module0_VP1-0_Matrix_Mask.csv",
        "trim": path + "Module0_VP1-0_Matrix_Trim.csv",
    },
    "VP2": {
        "trim0Mean": path + "Module0_VP2-0_Trim0_Noise_Mean.csv",
        "trimFMean": path + "Module0_VP2-0_TrimF_Noise_Mean.csv",
        "trim0Width": path + "Module0_VP2-0_Trim0_Noise_Width.csv",
        "trimFWidth": path + "Module0_VP2-0_TrimF_Noise_Width.csv",
        "mask": path + "Module0_VP2-0_Matrix_Mask.csv",
        "trim": path + "Module0_VP2-0_Matrix_Trim.csv",
    },
    "VP3": {
        "trim0Mean": path + "Module0_VP3-0_Trim0_Noise_Mean.csv",
        "trimFMean": path + "Module0_VP3-0_TrimF_Noise_Mean.csv",
        "trim0Width": path + "Module0_VP3-0_Trim0_Noise_Width.csv",
        "trimFWidth": path + "Module0_VP3-0_TrimF_Noise_Width.csv",
        "mask": path + "Module0_VP3-0_Matrix_Mask.csv",
        "trim": path + "Module0_VP3-0_Matrix_Trim.csv",
    },
}



# Function to load and reshape data for a given VP
def load_vp_data(files):
    # Load and reshape each dataset
    tMean0 = np.genfromtxt(files["trim0Mean"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tMeanF = np.genfromtxt(files["trimFMean"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tWidth0 = np.genfromtxt(files["trim0Width"], delimiter=",").astype(np.float16).reshape(256 * 256)
    tWidthF = np.genfromtxt(files["trimFWidth"], delimiter=",").astype(np.float16).reshape(256 * 256)
    mask = np.genfromtxt(files["mask"], delimiter=",").astype(np.float16).reshape(256 * 256)
    trim = np.genfromtxt(files["trim"], delimiter=",").reshape(256 * 256).astype(np.int8)
    
    # Combine inputs and outputs
    x = np.column_stack([tMean0, tMeanF, tWidth0, tWidthF, mask])  # Input features
    y = trim  # Labels
    return x, y

# Load data for all VPs
vp_data = {}
for vp, files in datasets.items():
    print(f"Loading data for {vp}...")
    vp_data[vp] = load_vp_data(files)

# Split data
x_vp0, y_vp0 = vp_data["VP0"]  
x_vp1, y_vp1 = vp_data["VP1"]  
x_vp2, y_vp2 = vp_data["VP2"]  
x_vp3, y_vp3 = vp_data["VP3"] 

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

    y = y/15

    return x, y

# Apply preprocessing to each dataset
x_vp0, y_vp0 = preprocess_data(x_vp0, y_vp0)
x_vp1, y_vp1 = preprocess_data(x_vp1, y_vp1)
x_vp2, y_vp2 = preprocess_data(x_vp2, y_vp2)
x_vp3, y_vp3 = preprocess_data(x_vp3, y_vp3)

# Get the length of the data and number of variables
dataLen = x_vp0.shape[0]
nVars = x_vp0.shape[1]
print(f"dataLen = {dataLen}, nVars = {nVars}")

# Split the data: even indices as training data
#x_data = x_vp1[::2]  # Even-indexed samples for training
#y_data = y_vp1[::2]  # Corresponding trim values for training
x_data = np.vstack((x_vp0, x_vp2))
y_data = np.hstack((y_vp0, y_vp2))

# Odd indices as evaluation data
#x_eval = x_vp1[1::2]  # Odd-indexed samples for evaluation
#y_eval = y_vp1[1::2]  # Corresponding trim values for evaluation
x_eval = np.vstack((x_vp1, x_vp3))
y_eval = np.hstack((y_vp1, y_vp3))

# Print shapes to verify the results
print(f"x_data shape after filtering: {x_data.shape}, y_data shape after filtering: {y_data.shape}")
print(f"x_eval shape after filtering: {x_eval.shape}, y_eval shape after filtering: {y_eval.shape}")


# Print data stats
print(f"Training data size: {x_vp0.shape}, {y_vp0.shape}")
print(f"VP1 evaluation data size: {x_vp1.shape}, {y_vp1.shape}")
print(f"VP2 evaluation data size: {x_vp2.shape}, {y_vp2.shape}")
print(f"VP3 evaluation data size: {x_vp3.shape}, {y_vp3.shape}")

nVal = x_data.shape[1]  # Number of input features
batch_size = 2 * 256
patience = 25
neurons = 2 * nVal
num_repeats = 3  # Number of times to repeat training for the same configuration

# Write results to a CSV file
csv_file = csvpath + "model_data2.csv"

# Define a single layer combination for testing multiple times
layer1, layer2 = 'relu', 'tanh' 


# Prepare results storage
results = []

# Loop to repeat the configuration multiple times
for i in range(num_repeats):
    print(f"Training iteration {i+1} for model with {layer1} & {layer2} layers...")

    # Define the model
    inputs = Input(shape=(nVal,))  # Adjust input shape based on x_data
    norm = BatchNormalization()(inputs)
    
    # First hidden layer
    layer = Dense(neurons, activation=layer1.lower())(norm)
    
    # Second hidden layer
    layer = Dense(neurons, activation=layer2.lower())(layer)
    
    # Output layer
    #outputs = Dense(16, activation='softmax')(layer)
    outputs = Dense(1, activation = 'linear')(layer)
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile the model
    model.compile(optimizer=keras.optimizers.get("Adam"),  # Adam optimizer
                  loss = keras.losses.MeanAbsoluteError(),  # Loss function: Regression loss mean squared error
                  metrics=[adjacent_accuracy]
                  )  # Metrics to track: accuracy
    
    # Define early stopping
    early_stopping = EarlyStopping(monitor="loss", patience=patience)
    
    # Measure training time
    start_time = time.time()
    history = model.fit(x=x_data, y=y_data,
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
        'Iteration': i + 1,  # Add iteration number for tracking
        'Input Layers': 2,
        'Layer Type': f"{layer1} & {layer2}",
        'Batch Size': batch_size,
        'Patience': patience,
        'Neurons': neurons,
        'Final Accuracy': final_accuracy,
        'Final Loss': final_loss,
        'Total Closeness': total_closeness,
        'Training Time (s)': training_time,
        #'Seed': seed,  # Log the seed used
        'Training data': 'VP0+2',
        'Evaluation data': 'VP1+3',
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
