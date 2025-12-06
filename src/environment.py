import gym # use the gym.Env class to create a custom environment (inheritance)
from gym import spaces # observational + action spaces for environment
import numpy as np # mainly arrays
import pyautogui # commit action
from screen_capture import capture_screen # to capture screen
import mss # for screen config
import time 

# output map:
# 0 - wait
# 1 - click
# 2 - hold
# 3 - release

import cv2
import numpy as np

def get_progress_percentage(frame):
    """
    frame: numpy array screenshot (H, W, C) in RGB or BGR
    returns a float between 0.0 and 1.0 representing level progress
    returns None if progress bar not found
    """
    # crop to region where the progress bar usually is
    # adjust these coordinates based on your screen / game settings
    bar_region = frame[0:50, 100:800]  # height 0-50, width 100-800
    
    # convert to grayscale
    gray = cv2.cvtColor(bar_region, cv2.COLOR_BGR2GRAY)
    
    # threshold to isolate filled part of bar
    # you may need to tune 200 based on actual bar color
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # assume largest contour is the filled part of the progress bar
    bar = max(contours, key=lambda c: cv2.contourArea(c))
    x, y, w, h = cv2.boundingRect(bar)
    
    # percentage filled = width of bar / total width of region
    full_width = bar_region.shape[1]
    percent = w / full_width
    return np.clip(percent, 0.0, 1.0)

class GDEnvironment(gym.Env): # inherits from the gym.Env class

    """an environment for the agent to play geometry dash"""

    def __init__(self):

        """config for the environment"""

        self.reward = 0 # initialize as 0, will go up or down based on model performance

        # death recognizer variables
        self.prev_frame = None
        self.prev_progress = 0.0
        self.no_progress_counter = 0
        self.reward = 0
        self.MAX_NO_PROGRESS = 10  # frames allowed without progress

        # config
        self.MAX_BUFFER = 20
        self.SHAPE = (300,463)

        # screen capture config - will be used in step() to update state
        self.sct = mss.mss() # create mss object 
        self.monitor = self.sct.monitors[1] # get screen index to capture 
        
        self.action_space = spaces.Discrete(4) # create action spaces -> jump, wait, hold, release
        # observation_space args:
        #   - shape -> resized array dimensions (image)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=self.SHAPE, dtype=np.float32) # create the observation space

        self.state = np.array([np.zeros(self.SHAPE) for _ in range(self.MAX_BUFFER)]) # 20 arrays -> all zeros
    
    def reset(self):

        """resets the environment state"""

        self.state = np.array([np.expand_dims(np.zeros(self.SHAPE), axis=-1)
                       for _ in range(self.MAX_BUFFER)], dtype=np.float32) # same is init, resets
        return self.state
    
    def step(self, action):
        """executes the specified action from the model and returns new state, reward, done, info"""

        done = False
        info = {}

        # perform action
        if action == 0:
            pass  # wait
        elif action == 1:
            pyautogui.leftClick()  # single click
        elif action == 2:
            pyautogui.mouseDown(button='left')  # hold
        elif action == 3:
            pyautogui.mouseUp(button='left')  # release

        # capture screen
        self.current_frame = capture_screen(sct=self.sct, monitor=self.monitor)
        self.current_frame = np.expand_dims(self.current_frame, axis=-1)  # shape (H, W, 1)

        # convert frame to uint8 for opencv
        frame_uint8 = (self.current_frame.squeeze() * 255).astype(np.uint8)
        rgb_frame = cv2.cvtColor(frame_uint8, cv2.COLOR_GRAY2BGR)

        # compute progress percentage
        progress = get_progress_percentage(rgb_frame)  # returns float 0-100

        # reward based on progress
        if not hasattr(self, 'prev_progress'):
            self.prev_progress = progress  # initialize first step

        # incremental reward: how much progress increased
        reward = max(progress - self.prev_progress, 0)  # only reward forward movement
        self.prev_progress = progress

        # death detection: if progress resets or extremely low for multiple frames
        if progress < 1.0:
            done = True
            reward = -30  # penalize death
            info['death'] = True
            pyautogui.leftClick()

        # update state buffer
        self.state = list(self.state)
        self.state.append(self.current_frame)
        del self.state[0]
        self.state = np.array(self.state, dtype=np.float32)

        return self.state, reward, done, info

    
    def render(self):
        pass

# testing

# import random

# time.sleep(3)

# # create the environment
# env = GDEnvironment()

# # reset to start a new episode
# state = env.reset()

# # loop for testing
# for step_num in range(1000):  # or however many steps you want
#     # pick a random action (0-3)
#     action = random.randint(0, 3)

#     # take step
#     state, reward, done, info = env.step(action)

#     # check if the player died
#     if info.get('death', False):
#         print(f"Step {step_num}: Player is dead!")
#         # reset the environment for next episode
#         state = env.reset()
#         env.prev_frame = None  # reset previous frame for brightness logic