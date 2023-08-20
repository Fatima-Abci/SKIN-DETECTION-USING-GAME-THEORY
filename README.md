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
<img src="Image URL" alt="Alt Text">
</p>
