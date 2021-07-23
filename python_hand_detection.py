from cv2 import cv2
import time
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

new_frame_time = 0
prev_frame_time = 0

cap = cv2.VideoCapture(0)  # Webcam feed

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
  while cap.isOpened():

    success, image = cap.read()
    if not success:
      print("Skipping Empty Frame !")

      continue
    

    image = cv2.flip(image, 1) # Comment this if webcam is mirrored
    
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #results = hands.process(image)

    hand = str(results.multi_handedness) # Detecting hand (Left or Right)
    if 'Right' in hand :
        whathand = 'Hand : Right'
    elif 'Left' in hand :
        whathand = 'Hand : Left'
    else :
        whathand = 'Hand : -'

    image.flags.writeable = False

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Calculating Frames Per Second (FPS)
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps2text = 'FPS : '+str(int(fps))

    # Displaying details
    cv2.rectangle(image,(5,5),(260,110),(0,170,240),-1)     
    cv2.putText(image, whathand, (20,50), cv2.FONT_HERSHEY_COMPLEX,1, (0,0,0), 2)   
    cv2.putText(image, fps2text, (20,90), cv2.FONT_HERSHEY_COMPLEX,1, (3,3,138), 2)   
    cv2.imshow('Hand Detection', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
      break
    
cap.release()
