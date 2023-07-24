import pyautogui
import time
import random

# Get the screen size
screen_width, screen_height = pyautogui.size()

# Run for 24 hours
end_time = time.time() + 24*60*60

while time.time() < end_time:
    # Generate a random position on the screen
    new_x = random.randint(0, screen_width)
    new_y = random.randint(0, screen_height)
    
    # Move the mouse to the new position
    pyautogui.moveTo(new_x, new_y, duration=1)  # Move the mouse to the new position over 1 second
    
    # Wait for 30 seconds before moving the mouse again
    time.sleep(30)
