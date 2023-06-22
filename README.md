# Open source pipeline for tree species prediction from drone footage


## Summary 

This repo provides the code for predicting trees using an orthomosiac from images captured using a drone.

The process is as follows: 

## Preprocessing

1. Rename images and put them all into a single folder  
2. Use ODM to build the orthomosaic  
3. Read in the orthomosaic and check each of the bands is visualised correctly  
4. Read in your file with your tree data and look at   
5. Read in the coordinates and check if they need to be recast to be on the same reference as the orthomosaic  
6. View the trees on the orthomosaic coloured by the species  

## Iterative prediction

Here we want to predict whether on a pixel level the trees can be predicted, for this step, we will iteratively increase the number of pixels around each tree, throwing away pixels that overlap. This means that different trees will have a different number of training data annotated to each one.

1. Annotate the surrounding pixels with the species  
2. Remove 25% of the trees as a validation set  
3. Run prediction on the pixels with different feature sets  
4. Test on validation set  
5. Repeat for 10 fold cross validation  


## Code directory
```
remhybmon
│   README.md
│   LICENSE
|   requirements.txt # Installation    
│
└───notes
│    |   scratch_notes.txt # Notes from any random 
|
│   
└───notebookes
|   │   PixelPredictionPipeline-waeldi-ubuntu.ipynb # the actual one that was run on the server
|   │   PixelPredictionPipeline-waeldi.ipynb # Prediction for RGB waeldi 
|   │   PixelPredictionPipeline-allenwiller.ipynb # Prediction for RGB allenwiller footage
|   │   Other notebooks # all other notebooks are scratch/working space 
|
└───data
│   └───public_data # Drone footage from RGB of allenwiller for public use
│       │   20210305_Beechdrought_GenotypedAdults.csv # CSV of coords (this is the file with the trees in it) 
│       │   allenwiller.tif # RGB orthomosaic of allenwiller
│       │   ...
```

## Methods

### Preprocessing 

### Orthomosaic

### Normalisation

### Data s

## References
1. OpenDroneMap Authors ODM – A command line toolkit to generate maps, point clouds, 3D models and DEMs from drone, balloon or kite images. OpenDroneMap/ODM GitHub Page 2020; https://github.com/OpenDroneMap/ODM
