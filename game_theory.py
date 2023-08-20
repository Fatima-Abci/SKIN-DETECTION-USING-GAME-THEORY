#===================================================================================================
#                              IMPORT THE NECESSARY LIBRARIES
#===================================================================================================
import numpy as np
import cv2 as cv
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from numpy import savetxt
import threading
from multiprocessing import Pool, cpu_count
from pathlib import Path
#===================================================================================================
#===================================================================================================


#===================================================================================================
#                                   FUNCTIONS
#===================================================================================================

#---------------------------------------------------------------------------------------------------

# convert an image to a CMYK color space
def rgb_to_cmyk(r, g, b):
    rgb_scale = 255
    cmyk_scale = 100
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale
    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)
    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy) 
    m = (m - min_cmy) 
    y = (y - min_cmy) 
    k = min_cmy
    # rescale to the range [0,cmyk_scale]
    return c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale

# rgb_to_cmyk_image function
def rgb_to_cmyk_image(org):
    cmyk_image = np.zeros((org.shape[0], org.shape[1], 4))
    for i in range(org.shape[0]):
        for j in range(org.shape[1]):
            r, g, b = org[i][j][::-1]
            c, m, y, k = rgb_to_cmyk(r, g, b) # Convert RGB to CMYK
            cmyk_image[i][j] = (c, m, y, k)
    return cmyk_image

#----------------------------------------------------------------------------------------------------

# init function
def init(img):
    mask_BGR = cv.inRange(img, (10, 10, 80), (100, 255, 170)) # mask BGR
    cmyk_img = rgb_to_cmyk_image(img) # CMYK image
    mask_cmyk = cv.inRange(cmyk_img, (0, 5, 5, 0), (7, 115, 200, 220)) # mask CMYK
    image = {
       "BGR" : img,
       "CMYK" : cmyk_img
    }
    maskTreshold_colorSpace = {
       "BGR" : mask_BGR,
       "CMYK" : mask_cmyk
    }
    color_spaces = {
        "HSV" : ((0, 15, 0), (17,170,255), cv.COLOR_BGR2HSV),
        "YCRCB" : ((0, 135, 85), (255, 180, 135), cv.COLOR_BGR2YCrCb),
        "LAB" : ((50, 138, 133), (255, 173, 255), cv.COLOR_BGR2LAB),
        "LUV" : ((30, 113, 100), (255, 160, 165), cv.COLOR_BGR2LUV)
    }
    for key, value in color_spaces.items():
        img_convert = cv.cvtColor(img, value[2])
        mask = cv.inRange(img_convert, value[0], value[1]) # retrieve the skin pixels
        image.update({key : img_convert})
        mask = cv.bitwise_not(mask)
        maskTreshold_colorSpace.update({key : mask}) 
    return image, maskTreshold_colorSpace

#----------------------------------------------------------------------------------------------------

# function that returns the confusion matrix of each color space on a specific region
def confMat(row_interval, col_interval, y_true_flat, maskTreshold_colorSpace):
    start_row = row_interval[0]
    end_row = row_interval[1]
    start_col = col_interval[0]
    end_col = col_interval[1]
    confmat = {}
    for key, value in maskTreshold_colorSpace.items():
        mask_flat = value[start_row : end_row, start_col : end_col].flatten()
        cm = confusion_matrix(y_true_flat, mask_flat)     
        confmat.update({key : cm})   
    return confmat

#----------------------------------------------------------------------------------------------------

# function that detects and classify homogeneous regions of an image 
# returns skin and non skin matrices
def homogeneous_regions(org, maskTreshold_colorSpace, y_true, block_size):
    skin = []
    nonSkin = []
    output_result = []
    # first loop (extract the homogeneous regions : Skin & Non skin regions)
    for i in range(0, org.shape[0], block_size):
        for j in range(0, org.shape[1], block_size):
            start_row = i
            end_row = i + block_size
            start_col = j
            end_col = j + block_size
            row_interval = [start_row, end_row]
            col_interval = [start_col, end_col]

            y_true_flat = y_true[start_row : end_row, start_col : end_col].flatten()
            cm = confMat(row_interval, col_interval, y_true_flat, maskTreshold_colorSpace)
            if(cm["HSV"].shape == (1, 1) or cm["YCRCB"].shape == (1, 1) 
              or cm["LAB"].shape == (1, 1) or cm["LUV"].shape == (1, 1)
              or cm["BGR"].shape == (1, 1) or cm["CMYK"].shape == (1, 1)):

                if(org[start_row : end_row, start_col : end_col].shape == (block_size, block_size, 3)):
                  region = org[start_row : end_row, start_col : end_col].flatten()

                  output_result.append((region, y_true_flat))

                if(y_true_flat[0] == 0):
                    for row in org[start_row : end_row, start_col : end_col]:
                        for pixel in row:
                            skin.append(pixel)
                elif(y_true_flat[0] == 255):
                    for row in org[start_row : end_row, start_col : end_col]:
                        for pixel in row:
                            nonSkin.append(pixel)
    # convert into numpy array
    skin = np.array(skin)
    nonSkin = np.array(nonSkin)
    # Reshape the array 
    skin = skin.reshape((skin.shape[0], 1, skin.shape[1]))
    nonSkin = nonSkin.reshape((nonSkin.shape[0], 1, nonSkin.shape[1]))
    return skin, nonSkin, output_result

#----------------------------------------------------------------------------------------------------

# generate the dataMatrix (skin and non skin) of each color space
def dataMatrix_colorSpace(skin, nonSkin):
    # generate DataMatrix for BGR
    dataMat_skin_bgr = np.reshape(skin, (-1, skin.shape[2]))    
    dataMat_nonSkin_bgr = np.reshape(nonSkin, (-1, nonSkin.shape[2]))
    # generate DataMatrix for CMYK
    skin_convert_cmyk = rgb_to_cmyk_image(skin) 
    nonSkin_convert_cmyk = rgb_to_cmyk_image(nonSkin) 
    dataMat_skin_cmyk = np.reshape(skin_convert_cmyk, (-1, skin_convert_cmyk.shape[2]))
    dataMat_nonSkin_cmyk = np.reshape(nonSkin_convert_cmyk, (-1, nonSkin_convert_cmyk.shape[2]))
    dataMatrix_skin = {
       "BGR" : dataMat_skin_bgr,
       "CMYK" : dataMat_skin_cmyk
    }
    dataMatrix_nonSkin = {
       "BGR" : dataMat_nonSkin_bgr,
       "CMYK" : dataMat_nonSkin_cmyk 
    }
    color_spaces = {
        "HSV" : cv.COLOR_BGR2HSV,
        "YCRCB" : cv.COLOR_BGR2YCrCb,
        "LAB" : cv.COLOR_BGR2LAB,
        "LUV" : cv.COLOR_BGR2LUV
    }
    for key, value in color_spaces.items():
        skin_convert = cv.cvtColor(skin, value)
        nonSkin_convert = cv.cvtColor(nonSkin, value)
        dataMat_skin = np.reshape(skin_convert, (-1, skin_convert.shape[2]))
        dataMat_nonSkin = np.reshape(nonSkin_convert, (-1, nonSkin_convert.shape[2]))
        dataMatrix_skin.update({key : dataMat_skin})
        dataMatrix_nonSkin.update({key : dataMat_nonSkin})
    return dataMatrix_skin, dataMatrix_nonSkin

#----------------------------------------------------------------------------------------------------

# dataMatrix_hybrid
def dataMatrix_hybrid(dataMatrix_skin, dataMatrix_nonskin, id_player):
  temp = id_player.copy()
  id_1 = temp.pop()
  id_2 = temp.pop()
  dataMatrix_skin_hybrid = np.hstack((dataMatrix_skin[id_1], dataMatrix_skin[id_2]))
  dataMatrix_nonskin_hybrid = np.hstack((dataMatrix_nonskin[id_1], dataMatrix_nonskin[id_2]))  
  while(len(temp) != 0):
    id = temp.pop()
    dataMatrix_skin_hybrid = np.hstack((dataMatrix_skin_hybrid, dataMatrix_skin[id]))
    dataMatrix_nonskin_hybrid = np.hstack((dataMatrix_nonskin_hybrid, dataMatrix_nonskin[id]))
  return dataMatrix_skin_hybrid, dataMatrix_nonskin_hybrid

#----------------------------------------------------------------------------------------------------

# function that returns the Mahalanobis distance between a pixel and a distribution
def calcul_distance(dataMatrix, x):
    M_dataMatrix = np.mean(dataMatrix, axis=0) # M vector (mean) 
    diff = x - M_dataMatrix # (x - m) term
    cov = np.cov(dataMatrix.T) # covariance matrix 
    #invcov = np.linalg.inv(cov) # inverse of the covariance matrix
    invcov = np.linalg.pinv(cov)
    #calculate the mahalanobis distance between the X vector and the M vector
    mdist = np.sqrt(np.dot(np.dot(diff, invcov), diff.T))
    return mdist

# mahalanobis_distance
def mahalanobis_distance(player, x_player):
    mdist_skin = calcul_distance(player["dataMatrix_skin"], x_player)
    mdist_nonskin = calcul_distance(player["dataMatrix_nonskin"], x_player)
    return mdist_skin, mdist_nonskin

#----------------------------------------------------------------------------------------------------

# function that defines a player 
def player_def(id_player, dataMatrix_skin, dataMatrix_nonskin, flag):
    temp = id_player.copy()
    if(flag == True):
      dataMatrix_skin_player, dataMatrix_nonskin_player = dataMatrix_hybrid(dataMatrix_skin, dataMatrix_nonskin, temp)
    else:
      id = temp.pop()
      dataMatrix_skin_player = dataMatrix_skin[id]
      dataMatrix_nonskin_player = dataMatrix_nonskin[id]
    player = {"id" : id_player, 
              "dataMatrix_skin" : dataMatrix_skin_player,
              "dataMatrix_nonskin" : dataMatrix_nonskin_player,
              "type" : flag
              }
    return player

#----------------------------------------------------------------------------------------------------

# function that updates a player (add mask)
def player_update(player, mask_player):
  player.update({"mask": mask_player})
  return player

#----------------------------------------------------------------------------------------------------

# x function
def x_vect(id_player, image_colorSpace, row_interval, col_interval, r, c):
   pixel = image_colorSpace[id_player][row_interval[0] : row_interval[1], col_interval[0] : col_interval[1]][r][c]
   x_player = np.array(pixel)
   return x_player

# generate the x_vector
def x_vector(player, image_colorSpace, row_interval, col_interval, r, c):
    id_player = player["id"].copy()
    id_prev = id_player.pop()
    x_player_prev = x_vect(id_prev, image_colorSpace, row_interval, col_interval, r, c)
    while(len(id_player) != 0):
      id_curr = id_player.pop()
      x_player_curr = x_vect(id_curr, image_colorSpace, row_interval, col_interval, r, c)
      x_player_prev = np.hstack((x_player_prev, x_player_curr))
    x_player = x_player_prev
    return x_player

#----------------------------------------------------------------------------------------------------

# classify a pixel as a skin or non skin pixel
def classify_pixel(d_nonskin_player, d_skin_player, mask_player, r, c):
  if(d_nonskin_player < d_skin_player):
    mask_player[r][c] = 255

#----------------------------------------------------------------------------------------------------

# returns the mean f1_score of a player
def f1_score_player(y_true, y_pred):
    f1_Score_0 = metrics.f1_score(y_true, y_pred, pos_label=0, zero_division = 0) # class 0
    f1_Score_255 = metrics.f1_score(y_true, y_pred, pos_label=255, zero_division = 0) # class 255
    mean_f1_score =  (f1_Score_0 + f1_Score_255)/2
    return mean_f1_score

#----------------------------------------------------------------------------------------------------

# returns the utility of a player
def utility_players(player_1, player_2, player_hybrid, y_true_flat):
    u_player_1 = f1_score_player(y_true_flat, player_1["mask"].flatten())
    u_player_2 = f1_score_player(y_true_flat, player_2["mask"].flatten())
    u_hybrid = f1_score_player(y_true_flat, player_hybrid["mask"].flatten())
    return u_player_1, u_player_2, u_hybrid 

#----------------------------------------------------------------------------------------------------

# round a floating-point value to 2 decimal 
def round_val(value):
    value = round(value, 2)
    return value

#----------------------------------------------------------------------------------------------------

# generate payoff matrix
def generate_payoffMatrix(u_player_1, u_player_2, u_hybrid):
  payoff_matrix = np.array([
      [(u_hybrid, u_hybrid), (u_hybrid, u_player_2)],
      [(u_player_1, u_hybrid), (u_player_1, u_player_2)]
  ])
  return payoff_matrix

#----------------------------------------------------------------------------------------------------

# equal_list
def equal_list(list):
    indices = []
    for i in range(len(list)):
        if(list[i] != list[0]):
            return []
        else:
            indices.append(i)
    return indices

#----------------------------------------------------------------------------------------------------

# function that returns the best responses of each player
def best_responses(payoff_matrix):
    BR1 = []
    BR2 = []
    for i in range(len(payoff_matrix)):
        indices1 = equal_list(payoff_matrix[:, i, 0])
        indices2 = equal_list(payoff_matrix[i, :, 1])
        if(indices1 == []):
            index = np.argmax(payoff_matrix[:, i, 0])
            BR1.append((index, i))
        else:
            for index in indices1:
                BR1.append((index, i))
        if(indices2 == []):
            index = np.argmax(payoff_matrix[i, :, 1])
            BR2.append((i, index))
        else:
            for index in indices2:
                BR2.append((i, index))
    return BR1, BR2

#----------------------------------------------------------------------------------------------------

# function that returns the nash equilibrium
def nash_equilibrium(BR1, BR2):
    NE = set(BR1) & set(BR2)
    return list(NE)
  
#----------------------------------------------------------------------------------------------------

# function winner
def winner(NE, player_1, player_2, hybrid_player, end_game, payoff_matrix):
    strategy = NE[0]
    gain = payoff_matrix[strategy[0], strategy[1]]
    payoff = np.max(gain)
    if len(NE) == 1:
        if gain[0] > gain[1]:
            winning_player = hybrid_player if strategy[0] == 0 else player_1
        elif gain[0] < gain[1]:
            winning_player = hybrid_player if strategy[1] == 0 else player_2
        else:
            if strategy[0] == 0 and strategy[1] == 0:
                winning_player = hybrid_player
            elif strategy[0] == 1 and strategy[1] == 0:
                winning_player = player_1
            else:
                winning_player = player_2
    elif len(NE) == 4:
        winning_player = player_2
    elif len(NE) == 2:
        winning_player = player_2 if gain[0] <= gain[1] else player_1

    if(payoff == 1):
        end_game = True  
    return winning_player, end_game

#----------------------------------------------------------------------------------------------------

def generate_mask(player_1, player_2, player_hybrid, block_size, row_interval, col_interval, round_game, image_colorSpace):
  if(round_game == 1):
    mask_player_1 = np.zeros((block_size, block_size))
  mask_player_2 = np.zeros((block_size, block_size))
  mask_hybrid = np.zeros((block_size, block_size))
  for r in range(block_size):
    for c in range(block_size):
        x_player_1 = x_vector(player_1, image_colorSpace, row_interval, col_interval, r, c) # X vector (current pixel)
        # generate a new mask for each player 
        if(round_game == 1):
          # player_1
          d_skin_player_1, d_nonskin_player_1 = mahalanobis_distance(player_1, x_player_1)
          classify_pixel(d_nonskin_player_1, d_skin_player_1, mask_player_1, r, c)
          
        # player_2
        x_player_2 = x_vector(player_2, image_colorSpace, row_interval, col_interval, r, c)
        d_skin_player_2, d_nonskin_player_2 = mahalanobis_distance(player_2, x_player_2)

        # hybrid
        x_hybrid = np.hstack((x_player_1, x_player_2))
        d_skin_hybrid, d_nonskin_hybrid = mahalanobis_distance(player_hybrid, x_hybrid)

        # classify skin & nonSkin pixels
        classify_pixel(d_nonskin_player_2, d_skin_player_2, mask_player_2, r, c)
        classify_pixel(d_nonskin_hybrid, d_skin_hybrid, mask_hybrid, r, c)
  # update players
  if(round_game == 1):
    player_1 = player_update(player_1, mask_player_1)
  player_2 = player_update(player_2, mask_player_2)
  player_hybrid = player_update(player_hybrid, mask_hybrid)
  return player_1, player_2, player_hybrid

#----------------------------------------------------------------------------------------------------

# function that determines the winner of the game
def determine_winner(player_1, player_2, player_hybrid, block_size, y_true_flat, end_game, row_interval, col_interval, round_game, image_colorSpace):
    player_1, player_2, player_hybrid = generate_mask(player_1, player_2, player_hybrid, block_size, row_interval, col_interval, round_game, image_colorSpace)
    u_player_1, u_player_2, u_hybrid = utility_players(player_1, player_2, player_hybrid, y_true_flat)
    payoff_matrix = generate_payoffMatrix(u_player_1, u_player_2, u_hybrid)
    print(payoff_matrix)
    print()
    BR1, BR2 = best_responses(payoff_matrix)
    list_NE = nash_equilibrium(BR1, BR2)
    winning_player, end_game = winner(list_NE, player_1, player_2, player_hybrid, end_game, payoff_matrix)
    return winning_player, end_game

#----------------------------------------------------------------------------------------------------

# game
def game(id_players, y_true_flat, dataMatrix_skin, dataMatrix_nonskin, block_size, row_interval, col_interval, image_colorSpace):
  round_game = 1
  end_game = False
  while(len(id_players) != 0 and end_game != True):
    if(round_game == 1):
      id_player_1 = id_players.pop()
      player_1 = player_def([id_player_1], dataMatrix_skin, dataMatrix_nonskin, False)
      id_player_2 = id_players.pop()
      player_2 = player_def([id_player_2], dataMatrix_skin, dataMatrix_nonskin, False)
      player_hybrid = player_def([id_player_1, id_player_2], dataMatrix_skin, dataMatrix_nonskin, True)
      print()
    else:
      player_1 = winner
      id = player_1["id"].copy()
      id_player_2 = id_players.pop()
      player_2 = player_def([id_player_2], dataMatrix_skin, dataMatrix_nonskin, False)
      id.append(id_player_2)
      player_hybrid = player_def(id, dataMatrix_skin, dataMatrix_nonskin, True)
    # game logic...
    winner, end_game = determine_winner(player_1, player_2, player_hybrid, block_size, y_true_flat, end_game, row_interval, col_interval, round_game, image_colorSpace)
    round_game = round_game + 1
  return winner

#----------------------------------------------------------------------------------------------------

# game theory approach
def game_theory(org, y_true_flat, dataMatrix_skin, dataMatrix_nonskin, block_size, row_interval, col_interval, image_colorSpace, output_result): 
    # list of id players
    id_players = ["CMYK", "BGR", "LAB", "LUV", "HSV", "YCRCB"]
    # winner
    winner = game(id_players, y_true_flat, dataMatrix_skin, dataMatrix_nonskin, block_size, row_interval, col_interval, image_colorSpace)
    region = org[row_interval[0] : row_interval[1], col_interval[0] : col_interval[1]].flatten()
    # Acquire the lock to protect access to results
    with threading.Lock():
        # Access the shared resource
        output_result.append((region, winner["mask"].flatten()))

#----------------------------------------------------------------------------------------------------

# second loop (apply game theory to the rest of the regions that are mixed ) 
# contains skin and non skin pixels, it's important to note that some of them may be homogeneous
def mixed_regions(org, maskTreshold_colorSpace, y_true, block_size, dataMatrix_skin, dataMatrix_nonskin, image_colorSpace, output_result):
    threads = [] # store threads
    for i in range(0, org.shape[0], block_size):
        for j in range(0, org.shape[1], block_size):
            start_row = i
            end_row = i + block_size
            start_col = j
            end_col = j + block_size
            row_interval = [start_row, end_row]
            col_interval = [start_col, end_col]

            y_true_flat = y_true[start_row : end_row, start_col : end_col].flatten()
            cm = confMat(row_interval, col_interval, y_true_flat, maskTreshold_colorSpace)

            if(cm["HSV"].shape == (2, 2) and cm["YCRCB"].shape == (2, 2)
                and cm["LAB"].shape == (2, 2) and cm["LUV"].shape == (2, 2)
                and cm["BGR"].shape == (2, 2) and cm["CMYK"].shape == (2, 2)):

                # ignore the regions with size < (16, 16, 3)
                if(org[start_row : end_row, start_col : end_col].shape == (block_size, block_size, 3)):
                    # execute game_theory in a separate thread
                    thread = threading.Thread(target = game_theory, args=(org, y_true_flat, dataMatrix_skin, dataMatrix_nonskin, block_size, row_interval, col_interval, image_colorSpace, output_result))
                    threads.append(thread)
                    thread.start()
    for thread in threads:  
        thread.join()
    return output_result

#----------------------------------------------------------------------------------------------------

# skin detection using game theory
def skin_detect_gameTheory(org, binary_grt, block_size):
    print("process will start his task (process an image)")
    # mask treshold and image of each color space
    image_colorSpace, maskTreshold_colorSpace = init(org)
   # retrieve homogeneous regions
    skin, nonSkin, output_result = homogeneous_regions(org, maskTreshold_colorSpace, binary_grt, block_size)
    # generate the dataMatrix for skin and nonSkin regions
    dataMatrix_skin, dataMatrix_nonskin = dataMatrix_colorSpace(skin, nonSkin)
    # apply game theory to mixed regions
    output_result = mixed_regions(org, maskTreshold_colorSpace, binary_grt, block_size, dataMatrix_skin, dataMatrix_nonskin, image_colorSpace, output_result)
    return output_result

#----------------------------------------------------------------------------------------------------

# process an image
def process_image(image_path):
    org_img_path = image_path[0]
    grt_img_path = image_path[1]
    org = cv.imread(str(org_img_path)) # read the original image
    org_median = cv.medianBlur(org, 5) # apply a median filter: (3,3) kernel
    binary_grt = cv.imread(str(grt_img_path), cv.IMREAD_GRAYSCALE) # ground truth mask
    output_result = skin_detect_gameTheory(org_median, binary_grt, block_size)
    print("process finished his task :D ")
    return output_result

#----------------------------------------------------------------------------------------------------

def fill_list(result):
    for elt in result:
        # Acquire the lock to protect access to regions and masks
        with threading.Lock():
        # Access the shared resource
            regions.append(elt[0])
            masks.append(elt[1])

#----------------------------------------------------------------------------------------------------

#====================================================================================================
#====================================================================================================


#====================================================================================================
#                                   IMPLEMENTATION
#====================================================================================================
# Load the HGR1 IMAGE TRAIN DATABASE 
org_img_directory = Path('path_to_trainset_org_img').glob('*') # load the original images
grt_img_directory = Path('path_to_testset_grt_img').glob('*') # load the ground truth images
block_size = 16 # define the block size
regions = [] # store each region of my image of size (block_size x block_size) (regions.csv)
masks = [] # store each mask of my image of size (block_size x block_size) (masks.csv)
threads_n = []


if __name__ == '__main__':
    num_processes = cpu_count()  # Get the number of CPU cores on your system
    pool = Pool(processes=num_processes)
    # Create a list of image paths
    image_paths = [(org_img_path, grt_img_path) for org_img_path, grt_img_path in zip(org_img_directory, grt_img_directory)]
    # Use the Pool.map method to process the images in parallel
    output_results = pool.map(process_image, image_paths)
    # Close the Pool to prevent any more tasks from being submitted
    pool.close()
    # Wait for all the processes to finish
    pool.join()

    for result in output_results:
        print("we are  spawnig threads to fill the regions and masks lists")
        thread = threading.Thread(target=fill_list, args=(result,))
        threads_n.append(thread)
        thread.start()
    for thread in threads_n:  
        thread.join()
        

#====================================================================================================
#====================================================================================================


#===================================================================================================
#                                   STORE THE RESULTS INTO A CSV FILE
#===================================================================================================
regions_arr = np.asarray(regions)
savetxt('regions.csv', regions_arr, delimiter=',')
masks_arr = np.asarray(masks)
savetxt('masks.csv', masks_arr, delimiter=',')
#====================================================================================================
#====================================================================================================