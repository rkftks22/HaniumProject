import cv2
import mediapipe as mp
import socket
import requests
import time 

# mediapipe 모델 정의
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

front_num=0

# JS
URL = ''
headers = {'Authorization' : ''}

# SOCKET 통신
HOST = '192.168.0.28'
PORT = 9999
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) # 소켓 객체 생성(IPv4, stream)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

# 통신 확인
try:
  server_socket.bind((HOST,PORT))
except socket.error:
  print("Bind Failed")

server_socket.listen() # 클라이언트의 접속 허용
client_socket, addr = server_socket.accept() # accpet에서 대기, 클라이언트 접속하면 새로운 소켓 리턴
print('Connected by', addr) # 접속한 클라이언트 주소 출력

# 카메라 
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
        continue

      try : #socket
        data = client_socket.recv(1024)
        if data.decode(): # receive data
              cv2.imwrite("capture.png", frame)
              image = cv2.imread("capture.png")
              
              # the BGR image to RGB.
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              # To improve performance, optionally mark the image as not writeable to
              # pass by reference.
              image.flags.writeable = False
              results = face_detection.process(image)

              
              image.flags.writeable = True
              image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
              if results.detections:
                for detection in results.detections:
                 mp_drawing.draw_detection(image, detection) # Localization
    
    
              if results.detections is not None:
                front_num = len(results.detections) # 식별 인원 수

              if data.decode() is not None: 
                print('Received from', data.decode())
                back_num = data.decode() # 수신받은 문자열 출력
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

client_socket.close()
server_socket.close()
cap.release() 
