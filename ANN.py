import numpy as np
import data_wrangling as data
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from sklearn.model_selection import train_test_split

timestamp = data.timestamp_array
Kp = data.Kp_array
X_train, X_test, Y_train, Y_test = train_test_split(timestamp, Kp, test_size=0.2, random_state=42)

# Input layers
F_input = Input(shape = timestamp.shape)      # Input for the dates

# LSTM layer for processing parameters
lstm_output = LSTM(64)(F_input)



# Additional layers for prediction
dense1 = Dense(32, activation='relu')(lstm_output)
output = Dense(2)(dense1)  # Assuming you're predicting a continuous value (e.g., future date)

# Create the model
model = keras.Model(inputs=F_input, outputs=output)

# Compile the model with an appropriate loss function and optimizer for your specific task
model.compile(optimizer='adam', loss='mean_squared_error')
print(model.summary())



# #training 
batch_size = 128 #the training and evaluation is done in batches of 64 making the process faster
epochs = 5

# history = model.fit(
#      x=X_train, # Training data
#      y=Y_train,                      # Target data for training
#      epochs=epochs,              # Number of training epochs
#      batch_size=batch_size,          # Batch size
#      verbose=0                       # Verbosity mode
#  )

#evaluate model
prediction_list =[]
for x_input in X_test:
    prediction = model.predict(X_test)
    prediction_list.append(prediction)

print("The Kp value is: ", prediction_list[0][0][1])


