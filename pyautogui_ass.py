
# send_on_click.py
import time
import threading
import pyautogui
from pynput import mouse, keyboard

# Configuration
MESSAGE = """`Hi  Team"""
SEND_ON = 'left'   # 'left' or 'right' (which mouse button triggers sending)
DEBOUNCE_SECONDS = 0.35  # ignore clicks that happen faster than this (prevents accidental double-sends)

# PyAutoGUI safety and timing
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.05

_last_send_time = 0.0
_stop_flag = False 

def send_message():

   
    time.sleep(0.08)
    pyautogui.typewrite(MESSAGE, interval=0.02)  
    pyautogui.press('enter')                    

def on_click(x, y, button, pressed):
    """
    Mouse callback. We act only on button press (not release).
    """
    global _last_send_time, _stop_flag
    if _stop_flag:
        return False  # stop listener if flagged

    if not pressed:
        return  # only handle press events

    # Only trigger for the configured button
    if button.name != SEND_ON:
        return

    now = time.time()
    if now - _last_send_time < DEBOUNCE_SECONDS:
        # ignore rapid successive clicks
        return
    _last_send_time = now

    # Run sending in a separate thread to avoid blocking the listener
    threading.Thread(target=send_message, daemon=True).start()

def on_press(key):
    """
    Keyboard callback. Press Esc to stop the program.
    """
    global _stop_flag
    try:
        if key == keyboard.Key.esc:
            print("Esc pressed — stopping listener.")
            _stop_flag = True
            # returning False from listener's stop method will be handled in main
            return False
    except AttributeError:
        pass

def main():
    print("== WhatsApp click-to-send script ==")
    print("1) Make sure WhatsApp (Web/Desktop) chat is open and the text input box is focused.")

    # Start keyboard listener (so Esc can stop the program)
    keyboard_listener = keyboard.Listener(on_press=on_press)
    keyboard_listener.start()

    # Start mouse listener (this will block until stopped)
    with mouse.Listener(on_click=on_click) as mouse_listener:
        # Keep running while listeners are alive and not flagged to stop
        while mouse_listener.running and not _stop_flag:
            time.sleep(0.1)

    keyboard_listener.stop()
    print("Program exited.")

if __name__ == "__main__":
    main()