import numpy as np
import cv2
height = 600
width = 800
color_buffer = np.zeros((height, width, 3), dtype=np.uint8)
color_buffer[:,:,2] = 254 # R
color_buffer[:,:,1] = 67  # G
color_buffer[:,:,0] = 101 # B
while True:
    cv2.imshow('pycuda', color_buffer)
    if (cv2.waitKey(1) & 0xFF) == 27: # esc退出程序
        break
exit()