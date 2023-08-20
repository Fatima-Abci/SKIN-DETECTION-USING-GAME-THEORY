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
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/b6116eb1-6297-4865-a698-4e606da478f6" alt="homogeneous region" width="200" height="200">
</p>

**Mixed region :** A mixed region is a region that contains both skin and non-skin pixels.
<p align="center"> 
<img src="https://github.com/FatimaAbc/SKIN-DETECTION-USING-GAME-THEORY/assets/66517563/b6116eb1-6297-4865-a698-4e606da478f6" alt="mixed region" width="200" height="200">
</p>
