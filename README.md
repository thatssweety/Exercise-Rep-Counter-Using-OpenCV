# Exercise-Rep-Counter-Using-OpenCV
![](https://github.com/thatssweety/Exercise-Rep-Counter-Using-OpenCV/blob/368f5f0efa786edf8f9b8275bf73cf52e5ef820c/rep%20count%20output1.gif)
## **Tech Stack Used** 

1. OpenCV
2. MediaPipe
3. Numpy

## Importing library and video image

```python

!pip install mediapipe
     

import cv2
from cv2 import destroyAllWindows
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
from google.colab.patches import cv2_imshow
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture("/content/drive/MyDrive/Knee bend/KneeBendVideo.mp4")

```
## Method for calculating angle

```python
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
  ```
  
  stretch_position=0;
bend_position=0;


## Rep Count Code
     
```python

counter = 0 
stage = None
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output_video_.mp4', fourcc, 24,(int(cap.get(3)), int(cap.get(4))))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is None:
            print('breaking')
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            break;
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            
            angle_knee = calculate_angle(hip, knee, ankle) #Knee joint angle
            angle_knee = round(angle_knee,2)
            
            knee_angle = 180-angle_knee
            
            
           
            cv2.putText(image, str(angle_knee), 
                           tuple(np.multiply(knee, [900, 800]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA
                                )
           
            if angle_knee > 165:
                stage = "up"
                stretch_position+=1;
                bend_position=0;
            if angle_knee <= 120 and stage =='up':
            stage="down"
                counter +=1
                print(counter)
                stretch_position=0;
                bend_position+=1;
            good_time = (1 / fps) * bend_position
            bad_time =  (1 / fps) * stretch_position
            if good_time < 8 and bad_time > 8:
              cv2.putText(image,str( 'Keep your knee bent'), (100,400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50,50,255), 2)    
        except:
            cv2.putText(image,str(counter), (400,400), cv2.FONT_HERSHEY_SIMPLEX, 2, (50,50,255), 2)
            counter=0;
            pass
        
        cv2.rectangle(image, (20,20), (250,120), (0,0,0), -1)
        cv2.putText(image, "Repetition : " + str(counter), 
                    (30,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
       
        cv2.putText(image, "Knee-joint angle : " + str(angle_knee), 
                    (30,100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(203,17,17), thickness=2, circle_radius=2) 
                                 ) 
        out.write(image)
        #cv2_imshow(image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
     
```
![](https://github.com/thatssweety/Exercise-Rep-Counter-Using-OpenCV/blob/368f5f0efa786edf8f9b8275bf73cf52e5ef820c/rep%20count%20output2.gif)
 
         


