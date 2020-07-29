# -*- coding: utf-8 -*-

import cv2
from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
def resize_image(image):
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(image)
    img_w, img_h = img.size
    M = max(img_w, img_h)
    background = Image.new('RGB', (M, M), (255, 255, 255))
    bg_w, bg_h = background.size
    offset = ((bg_w - img_w) / 2, (bg_h - img_h) / 2)
    offset = tuple(map(lambda x: int(x), offset))
    background.paste(img, offset)

    size = 224,224
    background = background.resize(size, Image.ANTIALIAS)
    imshow(background)
    background = np.array(background)
    #background = cv2.cvtColor(background, cv2.COLOR_RGB2BGR)
    return background



# while True:
#     image = cv2.imread('palm_template.jpg')
#     image = resize_image(image)
#     cv2.imshow('output', image)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()
