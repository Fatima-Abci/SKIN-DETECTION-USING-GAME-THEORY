# SKIN-DETECTION-USING-GAME-THEORY
This is my master's degree final project under the supervision of **Ms. Dahmani Djamila** and **Mr. Sahnoun Abdelkarim**.
## TABLE OF CONTENTS
- [OVERVIEW](#overview)
- [DESCRIPTION](#description)
  - [PREPROCESSING](#preprocessing)
  - [KEY CONCEPTS](#key-concepts)
  - [GAME THEORY PART](#game-theory-part)
  - [DEEP LEARNING PART](#deep-learning-part)
- [DATASET](#dataset)
- [RESULTS](#results)
- [RESSOURCES](#ressources)
## OVERVIEW
In this project, we have developed a skin detection model utilizing game theory. We have taken into account six color spaces as players in the game, namely HSV, YCRCB, LAB, CMYK, RGB, and LUV. Each player has the option to choose between two strategies: "collaborate" and "not collaborate." The gain of each player is represented by the F1-score metric.
## DESCRIPTION
We'll begin by presenting the preprocessing steps applied to the input image, and afterward, we'll explain the two parts of the project: the game theory section and the deep learning section.
### PREPROCESSING
We begin the process by reading an RGB image along with its corresponding ground truth. This ground truth image will be used to calculate the gain for each player. After this, we apply a median filter with dimensions of (5, 5) to the input image for noise reduction. It's important to mention that the input image is divided into smaller regions, each with dimensions of (16, 16, 3). This division allows us to treat our image per region rather than as a whole, aiming to achieve superior results as each region possesses its own characteristics.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/b6116eb1-6297-4865-a698-4e606da478f6" alt="devide image" width="200" height="200">
</p>

### KEY CONCEPTS
**Homogeneous region :** A homogeneous region is defined as a region composed exclusively of skin pixels or non-skin pixels. In other words, all the pixels present in this region are either skin pixels or something other than skin.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/5af8265b-3c90-463f-8ff3-648048b96f7e" alt="homogeneous region" width="600" height="200">
</p>

**Mixed region :** A mixed region is a region that contains both skin and non-skin pixels.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/6d11b7dd-505a-4f89-80f3-81a349bcc4e5" alt="mixed region" width="300" height="200">
</p>
Note that the size of the confusion matrix was used to distinguish between a homogeneous region and a mixed region. If the size of a confusion matrix is (1, 1), it means that only one class has been detected, indicating that the region is considered homogeneous. Otherwise, if it has a size of (2, 2), then two classes have been detected, and the region is considered mixed.

### GAME THEORY PART
**Extraction of homogeneous regions using thresholding** : This step involves detecting skin and non-skin regions in the input image by applying a simple thresholding technique to obtain the skin matrix and the non-skin matrix. These two matrices will be used in the subsequent part of this algorithm, specifically when applying game theory to mixed regions. The procedure we follow is as follows: for each region in our image, we compare the six color spaces mentioned earlier. Using their binary masks and the ground truth of the region, we calculate the confusion matrices and then check their sizes. If the size of the confusion matrix for one of the color spaces is equal to (1, 1), then the region is considered homogeneous. The pixels of homogeneous skin-type regions are stored in the skin matrix, while those of non-skin types are stored in the non-skin matrix. Furthermore, these homogeneous regions, each of size (16, 16, 3), will be saved along with their binary masks in two CSV files named "regions.csv" and "masks.csv," respectively. The purpose of these CSV files will be explained later in the section dedicated to deep learning.

**Apply game theory to mixed regions** : These regions were detected in the same way as the homogeneous regions, with the only difference being that the mixed region is an area where the confusion matrix size for all the color spaces used is equal to (2, 2). Now, let's proceed to explain the logic of the game. Initially, we have a list containing the six players we mentioned earlier. These players will compete in a two-by-two elimination-based tournament within each mixed region of our image. The game begins between the two color spaces: HSV and YCRCB. Each player has a choice between two strategies: "collaborate" or "not collaborate". 
- The **"not collaborate"** strategy allows a player to individually process the current region of the image. The processing involves calculating, for each pixel in a mixed region, its Mahalanobis distance with respect to the skin and non-skin data matrices. This results in two distances, which are then compared to classify the pixel as belonging to the skin class (0) or non-skin class (255).
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/19804019-ced7-462d-88ba-ff73b31ea078" alt="not collaborate strategy" width="800" height="400">
</p>

- The **"collaborate"** strategy, on the other hand, allows the two players to work together to process the current region by combining their color components, leading to the creation of a new hybrid player. The processing of the mixed region in this case proceeds in the same way as the "do not collaborate" strategy, which involves calculating the Mahalanobis distance per pixel.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/e4e28760-8fcd-45e3-84a8-f221fa0bc1da" alt="collaborate strategy" width="700" height="400">
</p>
At the end of this first tournament, we calculate the gain for each player, namely: HSV, YCRCB, and the new hybrid model, HSV-YCRCB. Next, we fill the payoff matrix, compute the Nash equilibrium within it, and then determine the winner. This winner will compete against another color space from the list of players, and this process will repeat until the list becomes empty, allowing us to iterate through all the color spaces. The corresponding mixed region and the binary mask of the winner will be saved in the two CSV files: "regions.csv" and "masks.csv," respectively."

### DEEP LEARNING PART
After generating the two CSV files containing the regions and corresponding binary masks for the training data of the game theory part, we are entering the phase of deep learning for our algorithm. This step involves training a convolutional autoencoder for content-based image retrieval using the k-neighrest neighbor algorithm.

**Training the autoencoder :** We start training the autoencoder with a dataset that has both skin and non-skin pixels. The following diagram represents the architecture of the utilized convolutional autoencoder.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/31e21ea3-045f-435c-ad14-ebc38f2525ce" alt="autoencoder" width="800" height="200">
</p>
The encoder part of the model allows for the extraction of the most important visual features from the input region.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/b609907a-a90c-45a3-811a-d7f8d6bbbc9c" alt="encoder" width="800" height="200">
</p>
The decoder, on the other hand, enables the reconstruction of the region from its representation in latent space.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/399a6b63-34f9-4abe-922c-77bbf42c3530" alt="decoder" width="800" height="200">
</p>
The training of the autoencoder allows for learning the most relevant representations of a given region of an image. It is essential to underline that our study focuses only on the encoder part of this model.

**Calculating similarity using the KNN algorithm :** Once the autoencoder has been trained and evaluated, we utilize the encoder component to obtain latent space representations of each region in the training dataset (regions.csv). These latent space representations are then used to train the k-nearest neighbors model. The logic of the final algorithm is as follows: for an input image, we divide it into fixed-size blocks (16, 16, 3) and process them individually. For each region, we generate its latent space and compute its similarity with the latent spaces of regions from the training dataset. By setting the parameter k to 1, we select the region that is most similar to it and apply its binary mask to it.

## DATASET
We worked with two databases from the [HGR](https://sun.aei.polsl.pl/~mkawulok/gestures/) series: HGR1 and HGR2A. These databases have been divided into training data (80%) and test data (20%).
## RESULTS
Here are some images from the HGR1 and HGR2A test databases, along with their corresponding binary masks generated by our skin detection model.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/12e8d89b-5d15-4d0c-87e3-1bb770b955c7" alt="result" width="600" height="250">
</p>

The table below presents the performance measurement values of our model.

| Dataset      | Precision      | Recall      | F1-score      | Accuracy      | False positive      |
|---------------|---------------|---------------|---------------|---------------|---------------|
| HGR1  | [0.98, 0.95]  | [0.98, 0.95]  | [0.98, 0.95]  | 0.97  | 0.05  |
| HGR2A  | [0.99, 0.87]  | [0.99, 0.86]  | [0.99, 0.86]  | 0.98  | 0.14  |

The results we have obtained for all the measures used demonstrate the robustness of our model for skin detection.
## RESSOURCES
- https://www.researchgate.net/publication/340845355_Zero-sum_game_theory_model_for_segmenting_skin_regions
- https://towardsdatascience.com/build-a-simple-image-retrieval-system-with-an-autoencoder-673a262b7921
