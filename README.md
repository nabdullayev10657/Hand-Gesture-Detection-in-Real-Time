# Hand-Gesture-Detection-in-Real-Time

This project provides real-time hand-gesture-detection with pretrained model available in this repository. The implementation has been done in both Tensorflow and PyTorch libraries.

For clonning:
```
git clone https://github.com/nabdullayev10657/Hand-Gesture-Detection-in-Real-Time.git
```

Prerequisite libraries:
```
pip install tensorflow
pip install torch
pip install numpy
pip install pyautogui
pip install opencv-python
```
You might need to use ```pip3``` if you face with error.

Here, both Tensorflow and Pytorch approaches has been added. For running Tensorflow file:
```
python tensorflow_model.py
```
For running Pytorch file:
```
python pytorch_model.py
```

Also, we should mention that ```tensorflow_model.py``` file uses ```handrecognition_model.h5``` file; however, ```pytorch_model.py``` file uses ```handrecognition_model.pth``` file that we have created in ```conversion.py``` file.

Additional References & More detailed information about ```handrecognition_model.h5``` architecture:

https://github.com/filipefborba/HandRecognition

https://github.com/omkarb09/Dino-game
