#!/usr/bin/env python
# coding: utf-8

# In[4]:


get_ipython().system('pip install opencv-python')


# In[1]:


import cv2

def cal_accum_avg(frame, accumulated_weight, background):
    if background is None:
        background = frame.copy().astype("float64")
        return background
    cv2.accumulateWeighted(frame, background, accumulated_weight)
    return background

def segment_hand(frame, threshold=25, background=None):
    if background is None:
        return None
    
    diff = cv2.absdiff(background.astype("uint8"), frame.astype("uint8"))
    _, thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return None
    else:
        hand_segment_max_cont = max(contours, key=cv2.contourArea)
        return (thresholded, hand_segment_max_cont)

cam = cv2.VideoCapture(0)

num_frames = 0
element = 'Z'
num_imgs_taken = 0
ROI_top = 100
ROI_bottom = 300
ROI_right = 150
ROI_left = 350
accumulated_weight = 0.5
background = None

while True:
    ret, frame = cam.read()

    frame = cv2.flip(frame, 1)

    frame_copy = frame.copy()

    roi = frame[ROI_top:ROI_bottom, ROI_right:ROI_left]

    gray_frame = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (9, 9), 0)

    color_frame = roi.copy()

    if num_frames < 156:
        background = cal_accum_avg(gray_frame, accumulated_weight, background)
        if num_frames <= 59:
            cv2.putText(frame_copy, "FETCHING BACKGROUND...PLEASE WAIT", (80, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    elif num_frames <= 300:
        hand = segment_hand(gray_frame, background=background)
        
        cv2.putText(frame_copy, "Adjust hand...Gesture for " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            
            cv2.putText(frame_copy, str(num_frames) + " For " + str(element), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Thresholded Hand Image", thresholded)
    
    else:
        hand = segment_hand(gray_frame, background=background)
        
        if hand is not None:
            thresholded, hand_segment = hand

            cv2.drawContours(frame_copy, [hand_segment + (ROI_right, ROI_top)], -1, (255, 0, 0), 1)
            
            cv2.putText(frame_copy, str(num_frames), (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame_copy, str(num_imgs_taken) + ' images' + " For " + str(element), (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Thresholded Hand Image", thresholded)
            
            if num_imgs_taken <= 156:
                cv2.imwrite(r"C:\Users\Krishna\Downloads\Gesture\train\\" + str(element) + "\\" + str(num_imgs_taken) + '.jpg', color_frame)
            else:
                break
            
            num_imgs_taken += 1
        else:
            cv2.putText(frame_copy, 'No hand detected...', (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.rectangle(frame_copy, (ROI_left, ROI_top), (ROI_right, ROI_bottom), (255, 128, 0), 3)
    
    cv2.putText(frame_copy, "Hand sign recognition_ _ _", (10, 20), cv2.FONT_ITALIC, 0.5, (51, 255, 51), 1)
    
    num_frames += 1

    cv2.imshow("Sign Detection", frame_copy)

    k = cv2.waitKey(1) & 0xFF

    if k == 27:
        break

cv2.destroyAllWindows()
cam.release()


# In[ ]:




