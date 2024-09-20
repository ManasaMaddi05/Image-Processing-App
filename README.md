# 📷 Image Processing App Project

## 🚀 Project Overview
This project implements a basic image processing app using Python. It progressively introduces different concepts such as image manipulation, object-oriented programming (OOP), and machine learning. By the end, the app can handle various image manipulations and includes a K-Nearest Neighbors (KNN) classifier for image label prediction.

## 📝 Key Parts

Part 1: RGB Image Representation
Part 2: Image Processing Template Class
Part 3: Standard Image Processing 
Part 4: Premium Image Processing (New Features)
Part 5: KNN Classifier for Image Label Prediction

## 🛠️ Technologies Used
Python: Core language
Pillow (PIL): For handling images (testing/viewing)
NumPy: For large data sets (testing)
tkinter: For viewing images

## 📂 Project Structure

1. RGBImage Class (Part 1)
Represents an image as a 3D list with RGB values. Key methods include:

__init__(): Initializes the image.
size(): Returns image dimensions.
get_pixels(): Returns a deep copy of the pixels.
copy(): Creates a copy of the image.
get_pixel(): Retrieves the pixel value at a position.
set_pixel(): Updates the pixel value at a position.
2. Image Processing Template Class (Part 2)
Implements image manipulation methods such as:

negate(): Inverts image colors.
grayscale(): Converts to grayscale.
rotate_180(): Rotates the image 180°.
get_average_brightness(): Calculates average brightness.
adjust_brightness(): Adjusts brightness by intensity.
blur(): Blurs the image by averaging neighbors.
3. Standard Image Processing Class (Part 3)

Features:
negate(), grayscale(), rotate_180(), adjust_brightness(), blur() (inherit from template but add cost).
redeem_coupon(): Reduces cost of future operations.
4. Premium Image Processing Class (Part 4)
Premium version with a fixed cost of $50. New methods:

chroma_key(): Replaces specific colors with background image pixels (like green screen).
sticker(): Places a smaller image on a background.
edge_highlight(): Highlights edges using convolution.
5. KNN Classifier (Part 5)
Machine learning classifier to predict image labels using the K-Nearest Neighbors (KNN) algorithm:

fit(): Stores training data.
distance(): Computes Euclidean distance between two images.
vote(): Determines the most frequent label among neighbors.
predict(): Predicts the label using the KNN algorithm.

## 📚 Key Concepts
Image Representation: Images are represented as 3D matrices with RGB values.
Deep vs. Shallow Copy: Ensures deep copies of images to prevent modifying the original.
KNN Classifier: Uses Euclidean distance to classify images based on their nearest neighbors.
