[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/OcE5Fe4c)

# Usage

As usual you can install the requirements via ```pip install -r requirements.txt```.<br>
**Be aware that it is very important to run this with python 3.11.x!**
<br>
<br>
If you have installed python 3.11 on your machine you can create a virtual environment with the 3.11 python interpreter via <br>```py -3.11 -m venv .venv```
<br>
<br>
The reason for this dependency is that tensorflow only works with python 3.12 or older and mediapie only works with python 3.11 or older. So presumably you can also run the code with older python version, but i haven not tested this!

Also the ```gesture_dataset_sample``` file needs to be copied into the assignment folder on the same level as the ```01-hyperparameters``` folder. Unfortunately this folder is to large, so i cannot push it to github.

## Hyperparameter Exlopartion

To see the results please look at the file ```./01-hyperparameters/hyperparameters_exploration.ipynb```
You can execute the code again using the right virtual environment or you can just take a look at the results. The approach and findings are inside the jupyter notebook either as results from functions or as markdown cells.
<br>
<br>
For some reason changing to python version 3.11 for mediapie hand recognition, the results changed quite a bit. But im not completly sure what the reason is. I hope this is allright and does not affect the grading.

## Dataset

You can find my recorded images and annotations under ```./02-dataset/jonas_images```. The code for the CNN training, predictions and confusion matrix is in ```./02-dataset/conf_matrix.ipynb```.
<br>

To get the boundin boxes in the right format i wrote a small script called ```extract_bbox.py```, there you can clit at the top left and bottom right of the bbox you want to extract. With the param ```--input_path``` you can specify the png/jpeg file you want to extract from. Results are printed to console. By cliking "Q" you cane exit the extraction script, by clicking "R" you can reset the bounding box to start over.
<br>
The confusion matrix is stored as ```conf-matrix.png```.

## Media Controls

To execute the gesture-based media controls script execute this command:<br>
```python .\03-media_control\media_control.py``` <br>
This then uses the pretrained model for predictions, i case you want to retrain the model you can run: <br>
```python .\03-media_control\media_control.py --train True```

As soon as you get a video feed from the webcam the predictions start and you can control e.g. a music track playing with windows media player. The following gestures work:

- like - volume up
- dislike - volume down
- stop - play / pause
- rock - next track

The volume will increase/decrease as long as your showing the guesture, for stop and rock you can toggle them by showing the guesture. For example if you want to stop the playing track you show the stop guesture. If you then want to continue the track you stop showing the guesture, wait for 1 second or so and then show it again. This cooldown is needed to prevent spamming the play/paus/next track input. 

The next track guesture is working fine, but make sure there is a next track available, by selecting a few songs for example and opening them together.

It is important that you show your hand for commands only, not because the no guesture is not handled but because the model sometimes interprets a moving hand as e.g. a like. I tried to solve this but i was not really able to. I guess its normal for a NN to also fail sometimes, it cant be perfect. In case you wonder how the no guesture is handled, it is used to implement the cooldown for the play/pause/nexttrack handling, so for each no gesture recognition a counter is counted up and new play/pause/nexttrack gestures can only be toggled once no gesture was recognized 10 times in a row.

You can exit the media control script py clicking "ESC" on your keyboard!


