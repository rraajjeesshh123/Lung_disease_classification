# Lung-disease-classification
The impact of lung diseases on public health is steadily increasing due to factors such as environmental changes, global warming, human behavior, and other causes. According to the WHO, more than four million premature deaths occur annually due to air pollution, encompassing conditions like asthma and pneumonia. The respiratory disorders range from slight discomfort and inconvenience by common colds and influenza viruses to life-threatening conditions like pneumonia, pneumothorax, and the recent pandemic of COVID-19.
Considering the importance and prevalence of Lung diseases worldwide, we trained an effective CNN-based multiclass classification model to classify Chest X-ray images of four GI diseases (lung opacity, COVID19, pneumonia, pneumothorax) and normal lung. The VGG19 CNN model achieved highest average accuracy of approximately 98.90% on the training set and accuracies of 99.50% and 99.60% on the validation and additional test set. 

## Package requirements:
* python = 3.9 
* tensorflow=2.11.0
* scikit-learn
* numpy
* pandas
* scipy

## Repository files:
The files contained in this repository are as follows:
* ``Readme file``: General information
* ``developed-model.h5``: Trained model
* ``prediction.py``: Main script to run prediction
* Folder named ``images``: Folder where images to be classified are to be saved

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
