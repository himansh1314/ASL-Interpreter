# ASL-Interpreter
Using CNN, American Sign Language is converted to text and speech. It is a useful tool for speech and hearing impaired. Works in real time, that too on embedded platform like Jetson Nano.

# Requirements
- Tensorflow-gpu 2+
- PIL
- OpenCV
- Numpy

# Content
- range_picker.py can be used to select HSV Upper and lower thresholds from an image. It may be useful so that you can get good values of segmentation.
- test_realtime.py is the file which has entire testing loop. It includes segmentation and model running part. Moreover,  GUI is provided the set the values of HSV thresholds.
- train.py - different models have been experimented. Mobilenet is found be the most efficient.
- utils.py contains a few necessary functions that are used by test_realtime.py.
- figures folders contains the training graphs of mobilenet on RMSProp and Adam Optimiser.
- models folder contains the tensorRT and .h5 model of mobilenet.
