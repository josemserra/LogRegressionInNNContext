# LogRegressionInNNContext
Binary Classification with Logistic Regression

Visual Studio (2015) project with for binay classification using logistic regression. First step to delve deeper into deep learning

## Step 1)
- Get tool to load images into c++ (CImg (http://cimg.eu/))
- Get tool for all the matrix operations required for dl (Eigen)
- Set VS project

Note: CImg (http://cimg.eu/) uses external libs to process the images. I installed this one and it solved the problems that I had ftp://ftp.graphicsmagick.org/pub/GraphicsMagick/windows/ (more specifically version GraphicsMagick-1.3.27-Q8-win64-dll, but I assume others will work)

## Step 2)
- Find dataset to test the project (Imagenet)
- Choose Images/Classes (dog vs non-dog)
- Download images and organise dataset
- Implement loading of images and creation of matrices on Eigen

## Step 3) 
- Neuron Initialization
- Forward Propagation
- Sigmoid Activation
- Loss and Cost Functions
	
## Step 4)
- Implement Backward Propagation

## Step 5)
- Implement Shuffle of Samples
- Implement (Batch) Gradient Descent

## Step 6) 
- Visualise Gradient Descent Steps using CImg

## Step 7)
- Predict output class
- Get Accuracy on train and dev sets
- Save and Load Eigen Matrices