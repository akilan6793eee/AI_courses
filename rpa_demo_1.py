import pyautogui  
import time
#mouse operation
pyautogui.click(100,100)
time.sleep(2)
pyautogui.rightClick(100,100)

time.sleep(10)
x,y=pyautogui.position()

print('x','y')

