
#=====================================================================================================
#                              IMPORT THE NECESSARY LIBRARIES
#=====================================================================================================
import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
import csv
#=====================================================================================================
#=====================================================================================================


#=====================================================================================================
#                                       IMPLEMENTATION
#=====================================================================================================
block_size = 16
X = np.loadtxt('path_to_train_csv_file', delimiter = ',') # load the train data (csv file)
X = X / 255.0 # normalize the data
X = X.reshape(len(X), block_size, block_size, 3) # reshape the data


# CREATE THE AUTOENCODER
input_img = Input(shape=(block_size, block_size, 3))
x = Conv2D(16,(3,3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2), padding='same')(x)
x = Conv2D(8,(3,3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2,2), padding='same', name='encoder')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding ='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse') # compile the autoencoder
print("autoencoder compiled")

# TRAIN THE AUTOENCODER MODEL
autoencoder.fit(X, X, epochs=2, batch_size=32, callbacks=None) # training
print("autoencoder trained")

# EVALUATE THE AUTOENCODER
mse_loss = autoencoder.evaluate(X, X, batch_size=32)
print("Mean Squared Error (MSE) Loss:", mse_loss)

autoencoder.save('autoencoder.h5') # save the autoencoder
autoencoder.summary() # display the architecture of the autoencoder model

# CREATE THE ENCODER PART
# the part that will encode the input into a latent space representation 
# (the dimension of this representation is 2, 2, 8)
encoder = Model(autoencoder.input, autoencoder.get_layer('encoder').output)
encoder.save('encoder.h5') # save the encoder
#=====================================================================================================
#=====================================================================================================


#=====================================================================================================
#                             STORE THE MSE OF THE AUTOENCODER INTO A CSV FILE
#=====================================================================================================

# Define the data to be stored in the CSV file
data = [
    ["mean squared error"],
    [mse_loss]
]

# Define the filename for the CSV file
filename = "mse_autoencoder.csv"

# Write data to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)


#=====================================================================================================
#=====================================================================================================