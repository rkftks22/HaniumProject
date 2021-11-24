import cv2
import mediapipe as mp
import socket
import requests
import time 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

front_num=0

# JS
URL = 'http://119.70.16.37:9002/api/car/123호1234'
headers = {'Authorization' : 'Basic YWRtaW46YWRtaW4'}

# SOCKET
HOST = '192.168.0.28'
PORT = 9999
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

try:
  server_socket.bind((HOST,PORT))
except socket.error:
  print("Bind Failed")

server_socket.listen()
client_socket, addr = server_socket.accept()
print('Connected by', addr)

cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FPS, 30)

def face_detection(cap):
  with mp_face_detection.FaceDetection(
      min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue

      try : #socket
        data = client_socket.recv(1024)
        if data.decode():
              cv2.imwrite("capture.png", frame)
              image = cv2.imread("capture.png")
              # Flip the image horizontally for a later selfie-view display, and convert
              # the BGR image to RGB.
              image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
              # To improve performance, optionally mark the image as not writeable to
              # pass by reference.
              image.flags.writeable = False
              results = face_detection.process(image)

              # Draw the face detection annotations on the image.
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
              if results.detections:
                for detection in results.detections:
                 mp_drawing.draw_detection(image, detection)
    
    
              if results.detections is not None:
                front_num = len(results.detections)

              if data.decode() is not None:
                print('Received from', data.decode())
                back_num = data.decode()    
                people_num = int(front_num) + int(back_num) 
                print("front_num : {}" .format(int(front_num)))
                print("back_num : {}" .format(int(back_num)))
                print("people num : {}" .format(people_num))
                db_data = {'cur_people' : people_num}
                requests.put(URL, json=db_data, headers=headers)
              
              data = 'NULL'
              #cv2.imshow('MediaPipe Face Detection', image)
          

      except : #socket
              if cv2.waitKey(30) & 0xFF == 27:
                break

face_detection(cap)
cur.close()
db.close()
cap.release() 