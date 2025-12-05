# captures screen contents of game

import mss # for screenshots
import numpy as np # for arrays
import cv2 as cv  # process data

def capture_screen(sct, monitor, target_height=200):

    """captures a screenshot of the gameplay and processes it"""

    raw_frame = sct.grab(monitor) # captures screen of selected monitor 
    img_array = np.array(raw_frame) # converts it into an image array
    img = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY) # converts from rgb to grayscale (faster processing)

    # preserve aspect ratio
    height, width = img.shape
    aspect_ratio = width / height
    new_width = int(target_height * aspect_ratio)

    img = cv.resize((new_width, target_height), img) # resize image

    return img