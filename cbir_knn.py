#=====================================================================================================
#                              IMPORT THE NECESSARY LIBRARIES
#=====================================================================================================
import numpy as np
from sklearn.neighbors import NearestNeighbors
from keras.models import load_model
from pathlib import Path
import cv2 as cv
from sklearn.metrics import confusion_matrix
import threading
import matplotlib.pyplot as plt
import csv
from sklearn.neighbors import BallTree
#=====================================================================================================
#=====================================================================================================


#=====================================================================================================
#                                          FUNCTIONS
#=====================================================================================================

#-----------------------------------------------------------------------------------------------------

# Function to calculate similarity for a given region
def mask_region(org, mask, row_interval, col_interval, encoder, encoder_shape, block_size):
  region = org[row_interval[0] : row_interval[1], col_interval[0] : col_interval[1]]
  size = region.shape # store the size of the region
  if(region.shape != (block_size, block_size, 3)):
    factors = np.ceil(np.array((block_size, block_size, 3)) / np.array(region.shape)).astype(int) # Calculate the required duplication factor for each dimension
    tiled_region = np.tile(region, factors) # Tile the region along each dimension
    tiled_region = tiled_region[:block_size, :block_size, :3] # Slice the tiled region to match the target size
    region = tiled_region
  query_region = region / 255.0 # normalize the query_region
  query_region = query_region.reshape(1,block_size, block_size, 3) # reshape the query_region into a 4D array
  query_code = encoder.predict(query_region) # generate the code of the query_region
  query_code = query_code.reshape(1, encoder_shape)  
  distances, indices = model.kneighbors(np.array(query_code)) #finds the nearest neighbors to the query_code
  # distances, indices = ball_tree.query(np.array(query_code), k=1)
  mask_similarReg_arr = masks[indices].reshape(-1, block_size, block_size) # reshape to (1, 8, 8)
  mask_similarReg = mask_similarReg_arr[0] # retrieve the mask of the similar region to query_region

  mask[row_interval[0] : row_interval[1], col_interval[0] : col_interval[1]] = mask_similarReg[:size[0], :size[1]]

#-----------------------------------------------------------------------------------------------------

# precision
def precision(tn, fp, fn, tp):
    precision_positive = tp / (tp + fp)
    precision_negative = tn / (tn + fn)
    return (precision_positive, precision_negative)

#-----------------------------------------------------------------------------------------------------

# recall
def recall(tn, fp, fn, tp):
    recall_positive = tp / (tp + fn)
    recall_negative = tn / (tn + fp)
    return (recall_positive, recall_negative)

#-----------------------------------------------------------------------------------------------------

# f1_score
def f1_score(precision_model, recall_model):
    f1_score_positive = (2 * (precision_model[0] * recall_model[0])) / (precision_model[0] + recall_model[0])
    f1_score_negative = (2 * (precision_model[1] * recall_model[1])) / (precision_model[1] + recall_model[1])
    return f1_score_positive, f1_score_negative

#-----------------------------------------------------------------------------------------------------

# accuracy
def accuracy(tn, fp, fn, tp):
    acc = (tn + tp) / (tn + fp + fn + tp)
    return acc

#-----------------------------------------------------------------------------------------------------

# false positive rate
def false_positive_rate(fp, tn):
    return fp / (fp + tn)


#=====================================================================================================
#=====================================================================================================


#=====================================================================================================
#                                   IMPLEMENTATION
#=====================================================================================================

block_size = 16
# Load the encoder model
encoder = load_model('encoder.h5')

encoder_shape =  2*2*8 # 2, 2, 8

masks = np.loadtxt('path_to_masks_csv_file', delimiter = ',') # load the masks.csv
regions = np.loadtxt('path_to_regions_csv_file', delimiter = ',') # load regions.csv
regions = regions / 255.0 # normalize the data
regions = regions.reshape(len(regions), block_size, block_size, 3) # reshape the data
print('generate the code of the regions(latent space representation)...')
codes = encoder.predict(regions) # generate the latent space representation of the regions
print("regions encoded...")
codes = codes.reshape(-1, encoder_shape) # reshape it (number_regions, 2*2*64) 
n_neigh = 1 # number of nearest neighbors to consider
model = NearestNeighbors(n_neighbors = n_neigh) # create an instance of NearestNeighbors
print('training the NearestNeighbors (model)...')
model = model.fit(codes) # fit the model
print('NearestNeighbors (model) trained...')
# Create a BallTree instance
# ball_tree = BallTree(codes)
# print("ballTree trained...")

# load the test dataset 
org_img_directory = Path('path_to_testset_org_img').glob('*') # load the original images (test set)
grt_img_directory = Path('path_to_testset_grt_img').glob('*') # load the ground truth images (test set)
# Create a list to store the threads
threads = []
precision_positive = 0
precision_negative = 0
recall_positive = 0
recall_negative = 0
f1_score_positive = 0
f1_score_negative = 0
accuracy_model = 0
fpr_model = 0
nbr = 0

#loop through the 2 folders in parallel using the zip function
for org_img_path, grt_img_path in zip(org_img_directory, grt_img_directory):
  org = cv.imread(str(org_img_path)) # read the original image
  org = cv.medianBlur(org, 5) # apply a median filter: (3,3) kernel
  grt = cv.imread(str(grt_img_path), cv.IMREAD_GRAYSCALE) # read the ground truth
  mask = np.zeros((org.shape[0], org.shape[1]))
  for i in range(0, org.shape[0], block_size):
     for j in range(0, org.shape[1], block_size):
      start_row = i
      end_row = i + block_size
      start_col = j
      end_col = j + block_size
      row_interval = [start_row, end_row]
      col_interval = [start_col, end_col]
      thread = threading.Thread(target = mask_region, args=(org, mask, row_interval, col_interval, encoder, encoder_shape, block_size))
      thread.start()
      threads.append(thread)  # Add the thread to the list

  # Wait for all threads to finish
  for thread in threads:  
    thread.join()  
  name = grt_img_path.name
  cv.imwrite(f"images/{name}", mask)
  #sys.exit()
  confMat = confusion_matrix(grt.flatten(), mask.flatten())
  tn, fp, fn, tp = confMat.ravel()
  prec = precision(tn, fp, fn, tp)
  precision_positive += prec[0]
  precision_negative += prec[1]

  rec = recall(tn, fp, fn, tp)
  recall_positive += rec[0]
  recall_negative += rec[1]

  acc = accuracy(tn, fp, fn, tp)
  accuracy_model += acc
  fpr = false_positive_rate(fp, tn)
  fpr_model += fpr
  nbr += 1

precision_positive = precision_positive / nbr
precision_negative = precision_negative / nbr
recall_positive = recall_positive / nbr
recall_negative = recall_negative / nbr
f1_score_positive, f1_score_negative = f1_score((precision_positive, precision_negative), (recall_positive, recall_negative))
accuracy_model = accuracy_model / nbr
fpr_model = fpr_model / nbr

#=====================================================================================================
#=====================================================================================================


#=====================================================================================================
#                                   STORE THE RESULTS INTO A CSV FILE
#=====================================================================================================

# Define the data to be stored in the CSV file
data = [
    ["precision", "recall", "f1_score", "accuracy", "false_positive_rate"],
    [[precision_positive, precision_negative], [recall_positive, recall_negative], [f1_score_positive, f1_score_negative], accuracy_model, fpr_model]
]

# Define the filename for the CSV file
filename = "metrics_simple.csv"

# Write data to the CSV file
with open(filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

#=====================================================================================================
#=====================================================================================================