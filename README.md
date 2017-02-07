# Facial Expression Recognition

A project for Advanced Machine Learning (Comp 6208) class in University of Southampton to predict facial expressions with machine learning algorithms.

##Team Members
Alex Young

Andreas Eliasson

Ara Hayrabedian

Lukas Weiss

Utku Ozbulak

##Data

The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

The training set consists of 28,709 examples. The first test set consists of 3,589 examples. The final test set consists of another 3,589 examples.

This dataset was prepared by Pierre-Luc Carrier and Aaron Courville, as part of an ongoing research project in the past. 

Past Kaggle competition for this partical challenge: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge

##To install
--On Linux
    Requires Boost lib

    Require X11 for gui support
         -sudo apt-get install libx11-dev

    dlib needs installing separably v 19.2
        - download from http://dlib.net/compile.html
        - go to root dir of dlib and type python setup.py install

    pip install -r requirements.txt
    
##Sample images


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

