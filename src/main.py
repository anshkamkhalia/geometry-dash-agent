# runs the agent

from screen_capture import capture_screen # to take a screenshot and convert into an array
import mss

# screen capture config 
sct = mss.mss() # create mss object 
monitor = sct.monitors[1] # get screen index to capture 

buffer = [] # create buffer frames to give model "memory"
MAX_BUFFER = 20 # saves previous 20 frames before deleting

while True:

    frame = capture_screen() # convert screen data to array
    break # temporary