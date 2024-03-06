# Using webcam without I/O ports
## Hardware setup
1. Plug an Arduino Uno to the PC used for recording.
2. Connect Arduino Uno digital pin 13 and GND to digital input of the recoding system (i.e. Open Ephys aquisition board)
3. Upload `onset_analogin.ino` to the Arduino (and also get the COM port number 'COM#')
4. Get the device ID for USB webcam (if only one camera is present, it is usually 0).
5. Run `webcam.py`