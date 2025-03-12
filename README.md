# Lung-disease-classification
The impact of lung diseases on public health is steadily increasing due to factors such as environmental changes, global warming, human behavior, and other causes. According to the WHO, more than four million premature deaths occur annually due to air pollution, encompassing conditions like asthma and pneumonia. Respiratory disorders range from slight discomfort caused by common colds and influenza viruses to life-threatening conditions like pneumonia, pneumothorax, and the recent pandemic of COVID-19.

Considering the importance and prevalence of lung diseases worldwide, we have trained an effective CNN-based multiclass classification model to classify chest X-ray images into five categories:

Normal lung
Lung opacity
COVID-19
Pneumonia
Pneumothorax
The VGG19 CNN model achieved the highest average accuracy of approximately 98.90% on the training set, with validation and additional test set accuracies of 99.50% and 99.60%, respectively.

## Package requirements:
* python = 3.9 
* tensorflow=2.11.0
* scikit-learn
* numpy
* pandas
* scipy

## Repository files:
The files contained in this repository are as follows:
* ``Readme file``: General information about the project
* ``developed-model.h5``: Pre-trained model for lung disease classification
* ``prediction.py``: Main script to run the prediction on input images
* Folder named ``images``: Folder where the chest X-ray images to be classified should be saved
* Folder named ``trainig_codes``:Contains the code for baseline and VGG19 models used for training
* ``baseline_model.py``: Contains the code for the baseline CNN model
* ``vgg19_model.py``: Contains the code for the VGG19 CNN model


## Usage:
> **_NOTE: _** Remember to activate the corresponding conda environment before running the script, if applicable.

**Input**: Image file (image.jpg)

**Output**: Prediction file

**Execution: **
**Step 1**: Install Anaconda3-5.2 or above.

**Step 2**: Install or upgrade libraries mentioned above (python, numpy, pandas, tensorflow, scikit-learn, scipy).

**Step 3**: Download and extract zipped file.

**Step 4**: 
Save the images to be predicted in "images" folder

**Step 5**: Change value of path variable in prediction_crc.py to the extracted folder and execute the script.

## Model Training:
To retrain or fine-tune the models, you can explore the training_codes folder. This folder contains the following training scripts:

Baseline CNN Model (baseline_model.py): A simple CNN architecture used as a starting point for classification.
VGG19 CNN Model (vgg19_model.py): A more advanced model based on the VGG19 architecture, which achieved superior performance in this project.
You can train these models on your own dataset by modifying the scripts or using them as-is. The models can be further fine-tuned, and training logs (accuracy, loss metrics) will be saved to the results folder.
