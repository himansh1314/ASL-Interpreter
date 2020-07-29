# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_vgg
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import utils
import time
model = load_model('models/vgg_final_lr001.h5')

test_gen = ImageDataGenerator(samplewise_center = True,
                               samplewise_std_normalization = True,
                               preprocessing_function = preprocess_vgg
                               )
def nothing(x):
    pass

def predict_class(cropped_image):
    """
    

    Parameters
    ----------
    cropped_image : Input image from segmentation
        This iamge is the image that we get after HSV colour space segmentation.
        Using this prediction on mobilenet can be made

    Returns
    -------
    prediction : Integer.
        Returns a number from 0 to 27.

    """
    img = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
    img = utils.resize_image(img)
    img = img.astype(np.float64)
    # Extra dimension for batch size
    img = np.expand_dims(img, axis = 0)
    # sample wise normalisation so that mean is 0
    img = test_gen.standardize(img)
    #give the prediction using predict_classes
    prediction = model.predict(img)
    prediction = np.argmax(prediction, axis = 1)
    return prediction
    
class_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "NOTHING",
    27: "SPACE"
    }

# Create a black image, a window
img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# create trackbars for color change
cv2.createTrackbar('min H','image',0,180,nothing)
cv2.createTrackbar('min S','image',0,255,nothing)
cv2.createTrackbar('min I','image',0,255,nothing)
cv2.createTrackbar('max H','image',0,180,nothing)
cv2.createTrackbar('max S','image',0,255,nothing)
cv2.createTrackbar('max I','image',0,255,nothing)

cv2.setTrackbarPos('min H','image',0) #25
cv2.setTrackbarPos('min S','image',0) #44
cv2.setTrackbarPos('min I','image',148) #95
cv2.setTrackbarPos('max H','image',179) #35
cv2.setTrackbarPos('max S','image',97) #126
cv2.setTrackbarPos('max I','image',255) #234
cap = cv2.VideoCapture(0)

while(True):
    # Read input from webcam and flip
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    stime = time.time()
    #frame = cv2.rectangle(frame, (100,100), (324,324), (0,255,0), thickness = 2)
    cv2.imshow('Input', frame)
    cv2.imshow('image', img)
    
    # get current positions of four trackbars
    minH = cv2.getTrackbarPos('min H', 'image')
    minS = cv2.getTrackbarPos('min S', 'image')
    minI = cv2.getTrackbarPos('min I', 'image')
    maxH = cv2.getTrackbarPos('max H', 'image')
    maxS = cv2.getTrackbarPos('max S', 'image')
    maxI = cv2.getTrackbarPos('max I', 'image')
    
    #display sample colour
    img[:, :256] = [minH, minS, minI]
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img[:,256:] = [maxH, maxS, maxI]
    #img2 = cv2.cvtColor(img2, cv2.COLOR_HSV2BGR)
    #Convert image to HSV and set upper and lower limit for segmentation
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_limit = (minH, minS, minI)
    upper_limit = (maxH, maxS, maxI)
    
    #Generate mask using upper and lower limit for segmentation
    mask = cv2.inRange(hsv_frame, lower_limit, upper_limit)
    # mask = cv2.GaussianBlur(mask, (5,5), 0)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((5,5),np.uint8))
    
    result = cv2.bitwise_and(hsv_frame, hsv_frame, mask = mask)
    bgr_frame = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
    bgr_frame = cv2.morphologyEx(bgr_frame, cv2.MORPH_CLOSE, np.ones((3,3),np.uint8))
    
    #Get contours from mask
    contour_image, contours, heirarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Getting the contours form black and white mask
    contours = sorted(contours, key = cv2.contourArea, reverse=True)
    cnt = contours[0] # Largest contour contains the end. We do this to eliminate noise from other contours
    cv2.drawContours(bgr_frame, [cnt], -1, (0,0,255), 3)
    
    #Get bounding box from largest contour
    x,y,w,h = cv2.boundingRect(cnt)
    
    # After getting the largest contour, get the image of hand
    hand_mask = np.zeros(bgr_frame.shape).astype(bgr_frame.dtype)
    cv2.fillPoly(hand_mask, [cnt], (255,255,255)) #Hand Mask now contains segmented image of hand. Black and white image
    hand_mask = cv2.bitwise_not(hand_mask)
    hand_image = cv2.bitwise_or(frame, hand_mask) #After bitwise_and, we now have hand mask which has original image of hand from the original frame. Segmentation done
    cv2.rectangle(hand_image, (x,y), (x+w, y+h), (255,0,0), 2)
    
    cropped_image = hand_image[y:y+h, x:x+w]
    predicted_class = predict_class(cropped_image)
    FPS = int(1/(time.time() - stime))
    cv2.putText(hand_image, class_dict[int(predicted_class)], (x+w, y+h), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 2, color = (255,0,0))
    cv2.putText(hand_image, "FPS: {}".format(FPS), (50, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 1, thickness = 2, color = (255,0,0))
    # Finally, display the windows
    cv2.imshow('segmented', bgr_frame)
    cv2.imshow('hand', hand_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
 
cv2.destroyAllWindows()
cap.release()
