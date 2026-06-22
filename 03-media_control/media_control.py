import mediapipe as mp
import mediapipe.python.solutions.hands as mp_hands_module

import argparse
import json
import os
import sys
import numpy as np
import cv2
import pyglet
from PIL import Image
from tqdm.notebook import tqdm
from pynput.keyboard import Controller, Key

from keras.models import Sequential, load_model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, RandomFlip, RandomContrast
from keras.metrics import categorical_crossentropy
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

CONDITIONS = ["like", "dislike", "stop", "rock"]
IMG_SIZE = 64
SIZE = (IMG_SIZE, IMG_SIZE)
COLOR_CHANNELS = 3
PATH = './gesture_dataset_sample'
LABELS_PATH = "03-media_control/label_names.json"

images = []
labels = []
label_names = []
annotations = dict()

# variables for hyperparameters
batch_size = 8 
epochs = 50 
num_classes = None
activation = 'relu'
activation_conv = 'leaky_relu'
layer_count = 2
num_neurons = 64

video_id = 0
cap = cv2.VideoCapture(video_id)

CAMERA_WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
CAMERA_HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Fallback if camer does not report a valid size
if CAMERA_WIDTH <= 0 or CAMERA_HEIGHT <= 0:
    CAMERA_WIDTH, CAMERA_HEIGHT = 640, 480
WINDOW_WIDTH, WINDOW_HEIGHT = CAMERA_WIDTH + 200, CAMERA_HEIGHT + 400
window = pyglet.window.Window(WINDOW_WIDTH, WINDOW_HEIGHT)
exit_text = pyglet.text.Label("Press ESC to exit", font_size=18, x=10, y=10)

# initialize mediapipe hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# initialize pynput controller
controller = Controller()
current_action = None
no_action_count = 0

# converts OpenCV image to PIL image and then to pyglet texture
# https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55
def cv2glet(img,fmt):
    '''Assumes image is in BGR color space. Returns a pyimg object'''
    if fmt == 'GRAY':
      rows, cols = img.shape
      channels = 1
    else:
      rows, cols, channels = img.shape

    raw_img = Image.fromarray(img).tobytes()

    top_to_bottom_flag = -1
    bytes_per_row = channels*cols
    pyimg = pyglet.image.ImageData(width=cols, 
                                   height=rows, 
                                   fmt=fmt, 
                                   data=raw_img, 
                                   pitch=top_to_bottom_flag*bytes_per_row)
    return pyimg

# Copied from hyperparameters.ipynb
def preprocess_image(img):
    if COLOR_CHANNELS == 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img, SIZE)
    return img_resized

# Copied from hyperparameters.ipynb
def load_images_and_annotataions():
    for condition in CONDITIONS:
        with open(f'{PATH}/_annotations/{condition}.json') as f:
            annotations[condition] = json.load(f)
            
    for condition in CONDITIONS:
        for filename in tqdm(os.listdir(f'{PATH}/{condition}')):
            # extract unique ID from file name
            UID = filename.split('.')[0]
            img = cv2.imread(f'{PATH}/{condition}/{filename}')
            
            # get annotation from the dict we loaded earlier
            try:
                annotation = annotations[condition][UID]
            except Exception as e:
                print(e)
                continue
            
            # iterate over all hands annotated in the image
            for i, bbox in enumerate(annotation['bboxes']):
                # annotated bounding boxes are in the range from 0 to 1
                # therefore we have to scale them to the image size
                x1 = int(bbox[0] * img.shape[1])
                y1 = int(bbox[1] * img.shape[0])
                w = int(bbox[2] * img.shape[1])
                h = int(bbox[3] * img.shape[0])
                x2 = x1 + w
                y2 = y1 + h
                
                # crop image to the bounding box and apply pre-processing
                crop = img[y1:y2, x1:x2]
                preprocessed = preprocess_image(crop)
                
                # get the annotated hand's label
                # if we have not seen this label yet, add it to the list of labels
                label = annotation['labels'][i]
                if label not in label_names:
                    label_names.append(label)
                
                label_index = label_names.index(label)
                
                images.append(preprocessed)
                labels.append(label_index)
          
# Load label names from stores json      
def load_label_names():
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Copied from hyperparameters.ipynb
def train_classifier():
    load_images_and_annotataions()
    num_classes = len(label_names)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    # Convert to float
    X_train = np.array(X_train).astype('float32')
    # Normalization [0,1]
    X_train = X_train / 255.

    X_test = np.array(X_test).astype('float32')
    X_test = X_test / 255.

    # Convert interger class labels in to one hot vector
    y_train_one_hot = to_categorical(y_train)
    y_test_one_hot = to_categorical(y_test)

    # rename for demonstration purposes
    train_label = y_train_one_hot
    test_label = y_test_one_hot

    X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
    X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS)
    
    # define model structure
    # with keras, we can use a model's add() function to add layers to the network one by one
    model = Sequential()

    # data augmentation (this can also be done beforehand - but don't augment the test dataset!)
    model.add(RandomFlip('horizontal'))
    model.add(RandomContrast(0.1))

    # first, we add some convolution layers followed by max pooling
    model.add(Conv2D(64, kernel_size=(9, 9), activation=activation_conv, input_shape=(SIZE[0], SIZE[1], COLOR_CHANNELS), padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), padding='same'))

    model.add(Conv2D(32, (5, 5), activation=activation_conv, padding='same'))
    model.add(MaxPooling2D(pool_size=(3, 3), padding='same'))

    model.add(Conv2D(32, (3, 3), activation=activation_conv, padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Randomly disables 20% of neurons during each training step
    model.add(Dropout(0.2))

    # after the convolution layers, we have to flatten the data so it can be fed into fully connected layers
    model.add(Flatten())

    # add some fully connected layers ("Dense")
    for i in range(layer_count - 1):
        model.add(Dense(num_neurons, activation=activation))

    model.add(Dense(num_neurons, activation=activation))

    # for classification, the last layer has to use the softmax activation function, which gives us probabilities for each category
    model.add(Dense(num_classes, activation='softmax'))

    # specify loss function, optimizer and evaluation metrics
    # for classification, categorial crossentropy is used as a loss function
    # use the adam optimizer unless you have a good reason not to
    model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=['accuracy'])

    # define callback functions that react to the model's behavior during training
    # in this example, we reduce the learning rate once we get stuck and early stopping
    # to cancel the training if there are no improvements for a certain amount of epochs

    # If validation loss stops improving for 2 epochs, multiply  the learning rate by 0.2 (slow down/fine tune)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
    
    # if validation loss doesn't improve for 3 epchs, stop training entierly
    stop_early = EarlyStopping(monitor='val_loss', patience=3)
    
    # train the model
    history = model.fit(
        X_train,
        train_label,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, test_label),
        callbacks=[reduce_lr, stop_early]
    )
    
    # store the trained model
    model.save("03-media_control/gesture_classifier.keras")
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(label_names, f)
    return model

# predict the sign from the cnn ready crop
def predict_sign(ai_crop):
    probs = model.predict(ai_crop, verbose=0)[0]
    class_index = int(np.argmax(probs))
    return label_names[class_index], probs[class_index]

def crop_hand(frame):
    h, w = frame.shape[:2]
    
    # Convert the image to RGB as Mediapipe expects RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # If a hand is detected, crop to the bounding box of the hand with some padding.
    if results.multi_hand_landmarks:
        # Get hand bounding box from landmarks
        landmarks = results.multi_hand_landmarks[0]
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        # Padding based on the size of the bounding box
        box_w = x_max - x_min
        box_h = y_max - y_min
        # Padding is 10% of the bounding box
        pad_x = box_w * 0.1
        pad_y = box_h * 0.1
        
        x_min = max(0, int((x_min - pad_x) * w))
        y_min = max(0, int((y_min - pad_y) * h))
        x_max = min(w, int((x_max + pad_x) * w))
        y_max = min(h, int((y_max + pad_y) * h))
        
        # Crop hand region
        crop = frame[y_min:y_max, x_min:x_max]
    else:
        # If no hand is detected, return a blank image
        crop = np.zeros((SIZE[0], SIZE[1], COLOR_CHANNELS), dtype=np.uint8)
    
    # Preprocess the crop for the model
    crop = preprocess_image(crop)
    
    # Normalize and format for the model
    crop = crop.astype("float32") / 255.0
    crop = np.expand_dims(crop, axis=0) 
    
    if len(crop.shape) == 3 and COLOR_CHANNELS == 1:
        crop = np.expand_dims(crop, axis=-1)

    return crop

def toggle_pynput_action(label):
    global current_action, no_action_count
    if label in CONDITIONS:
        
        # If like turn volume up
        if label == "like":
            no_action_count = 0
            controller.press(Key.media_volume_up)
            controller.release(Key.media_volume_up)
            current_action = "volume_up"
            print("Volume Up")
            
        # If dislike turn volume down
        if label == "dislike":
            no_action_count = 0
            controller.press(Key.media_volume_down)
            controller.release(Key.media_volume_down)
            current_action = "volume_down"
            print("Volume Down")
        
        # If stop toggle play/pause but only if we haven't already triggered play/pause in the last 10 frames
        if label == "stop" and no_action_count > 10:
            no_action_count = 0
            if current_action != "play_pause":
                controller.press(Key.media_play_pause)
                controller.release(Key.media_play_pause)
            current_action = "play_pause"
            print("Play/Pause")
        
        # If rock skip to the next track but only if we haven't already triggered next track in the last 10 frames
        if label == "rock" and no_action_count > 10:
            no_action_count = 0
            if current_action != "next_track":
                controller.press(Key.media_next)
                controller.release(Key.media_next)
            current_action = "next_track"
            print("Next Track")
    
    # If no action is detected, increase the no_action_count. This allows cooldown for actions.
    else:
        current_action = "no_action"
        no_action_count += 1
        
@window.event
def on_key_press(symbol, modifiers):
    # Exit the application when the ESC key is pressed
    if symbol == pyglet.window.key.ESCAPE:
        cap.release()
        cv2.destroyAllWindows()
        pyglet.app.exit()
    

@window.event
def on_draw():
    window.clear()
    exit_text.draw()
    
    ret, frame = cap.read()
    if not ret:
        return

    # Crop the hand using mediapipe and preprocess it for the model
    ai_crop = crop_hand(frame)
    
    # Predict the sign from the cropped image
    label, confidence = predict_sign(ai_crop)
    print(label, confidence)
    
    toggle_pynput_action(label)

    # Draw the main video feed
    img = cv2glet(frame, 'BGR')
    img.blit(100, 100)

    # Convert the AI crop back to uint8 for visualization
    visual_crop = (ai_crop[0] * 255).astype(np.uint8)
    
    # If the crop is grayscale, convert it back to BGR for visualization
    visual_crop = cv2.resize(visual_crop, (200, 200))

    # Draw the cropped hand region
    crop_img = cv2glet(visual_crop, 'BGR')
    crop_img.blit(100, CAMERA_HEIGHT + 120)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier on the gesture dataset")
    parser.add_argument("--train", type=bool, default=False, help="Whether to train the classifier")
    parser.add_argument("--video_id", type=int, default=0)
    args = parser.parse_args()

    if args.train:
        model = train_classifier()
    else:
        label_names = load_label_names()
        model = load_model("03-media_control/gesture_classifier.keras")
    
    try:
        pyglet.app.run()
    except KeyboardInterrupt:
        cap.release()
        cv2.destroyAllWindows()
        pyglet.app.exit()