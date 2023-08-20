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
## DATASET
We worked with two databases from the [HGR](https://sun.aei.polsl.pl/~mkawulok/gestures/) series: HGR1 and HGR2A. These databases have been divided into training data (80%) and test data (20%).
## RESULTS
## RESSOURCES
- https://www.researchgate.net/publication/340845355_Zero-sum_game_theory_model_for_segmenting_skin_regions
- https://towardsdatascience.com/build-a-simple-image-retrieval-system-with-an-autoencoder-673a262b7921
