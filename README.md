# Facial Expression Recognition

**Note**: This is a messy repository, you should find your way through what is present here (unfortunately).

## Team Members

* Alex Young
* Andreas Eliasson
* Ara Hayrabedian
* Lukas Weiss
* Utku Ozbulak

## Data

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The training set consists of 28,709 examples. The first test set consists of 3,589 examples. The final test set consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project in the past. 

Past Kaggle competition for this partical challenge: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge


![Angry](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/0_anrgy/10.png "Angry")
Angry

![Disgust](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/1_disgust/299.png "Disgust")
Disgust

![Fear](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/2_fear/5.png "Fear")
Fear

![Happy](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/3_happy/14.png "Happy")
Happy

![Sad](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/4_sad/6.png "Sad")
Sad

![Surprise](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/5_suprise/15.png "Surprise")
Surprise

![Neutral](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/6_neutral/11.png "Neutral")
Neutral

## Evaluated Models

#### Facial Landscape - Pixel Approach
~50% Accuracy on detected faces with SVMs

*SVM Optimisation:*


![SVM](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/svm.png "SVM_opt")


#### Facial Landscape - Vector Approach
~49% Accuracy on detected faces with neural nets

#### SIFT
~28% Accuracy on detected faces with SVMs

#### Convolutional Neural Networks
~51% Accuracy on all data (7 classes)

*CNN Model Breakdowns:*

![CNN](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/cnn.png "CNN")

#### Pre-trained Models
~49% Accuracy with bagged model

![Pre](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/pre.png "Pre")

#### Final Results
![Final](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/final.png "Final")


## Top Two Predictions with the Best Model
![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_andreas.png "bw")
**Supposed to be:** Surpsised **Found:**  93% Neutral - 3% Sad

![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_alex.png "bw")
**Supposed to be:** Angry **Found:**  43% Sad - 21% Angry

![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_ara.png "bw")
**Supposed to be:** Happy **Found:**  86% Happy - 11% Neutral

![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_charles.png "bw")
**Supposed to be:** Neutral **Found:**  83% Neutral - 7% Angry

![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_lukas.png "bw")
**Supposed to be:** Sad **Found:**  33% Sad - 25% Fear

![bw](https://raw.githubusercontent.com/utkuozbulak/facial-expression-recognition/master/data/samples/bw_utku.png "bw")
**Supposed to be:** Fear **Found:**  94% Neutral - 4% Fear
