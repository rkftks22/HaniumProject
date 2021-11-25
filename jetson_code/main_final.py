import mediapipe as mp
import socket
import dlib, cv2
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
import threading
from functools import wraps
from scipy.spatial import distance
import RPi.GPIO as GPIO

#model
#dlib model
detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("./models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1('./models/dlib_face_recognition_resnet_model_v1.dat')
#mediapipe model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# JS
headers = {'Authorization' : ''}
URL_server = ''
URL_web = ''
URL_web_user = URL_web +''
URL_web_reservation = URL_web + ''

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

# data
data = {}
img_paths = {}
descs = {}
lastsave = 0
cursave = 0 
output_pin = 18 #buzzer

#counting
def counter(func):
    @wraps(func)
    def tmp(*args, **kwargs):
        tmp.count += 1
        time.sleep(0.05)
        global lastsave
        global cursave
        cursave = time.time()
        if cursave - lastsave > 5:
            lastsave = time.time()
            tmp.count = 0
        return func(*args, **kwargs)
    tmp.count = 0
    return tmp

@counter
def close():
    print('eye closed')

def sound():
    GPIO.output(output_pin, GPIO.HIGH)
    time.sleep(2) 
    GPIO.output(output_pin, GPIO.LOW)

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

# url image load
def img_load(url):
    image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    return img

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = dlib_facelandmark(img, d)
        
        # convert dlib shape to numpy array
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

for a in range(len(requests.get(URL_web_user, headers = headers).json()['_embedded']['userResourceList'])):
    img_url = URL_server+'image/uploads/'+str(a+1)+'.jpg'
    img_paths[a+1] = img_url
    descs[a+1] = None

for name, img_path in img_paths.items():
    img_bgr = img_load(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]


np.save('./img/descs.npy', descs)

def cam1():
    user_name = 0
    car_id = 0
    global output_pin
    URL_recog = ''
    certification_name = ""
    
    # video setting
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    
    # gpio setting
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(output_pin, GPIO.OUT, initial = GPIO.LOW)
    
    # face_recognition
    while True:
        for i in range(len(requests.get(URL_web_reservation, headers = headers).json()['_embedded']['reservationResourceList'])):
            print(f'num : {i}')
            recog = requests.get(URL_web_reservation, headers = headers).json()['_embedded']['reservationResourceList'][i]['certification']
            print(f'recog : {recog}') # check recognition
            if(recog == 'DOING'):
                user_name = requests.get(URL_web_reservation, headers = headers).json()['_embedded']['reservationResourceList'][i]['userId']
                car_id = requests.get(URL_web_reservation, headers = headers).json()['_embedded']['reservationResourceList'][i]['reservationId']
                print(f'user name : {user_name} / car id = {car_id}') # check user_name, car_id 
                ret,frame = camera.read()
                cv2.imwrite("recog.png", frame)
                print("capture")

                img_bgr = cv2.imread('recog.png')
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                rects, shapes, _ = find_faces(img_rgb)
                descriptors = encode_faces(img_rgb, shapes)
                # Find the same face
                for i, desc in enumerate(descriptors):
                    found = False
                    for name, saved_desc in descs.items():
                        dist = np.linalg.norm([desc] - saved_desc, axis=1)

                        if dist < 0.6:
                            found = True
                            rect = patches.Rectangle(rects[i][0],
                                                rects[i][1][1] - rects[i][0][1],
                                                rects[i][1][0] - rects[i][0][0],
                                                linewidth=2, edgecolor='w', facecolor='none')
                            certification_name=name
                            print(f'name : {certification_name}') # Name of the face that matches the trained data. 
                            break
                        if not found:
                            rect = patches.Rectangle(rects[i][0],
                                                rects[i][1][1] - rects[i][0][1],
                                                rects[i][1][0] - rects[i][0][0],
                                                linewidth=2, edgecolor='r', facecolor='none')
                            print('unknown')
                            
                if user_name == certification_name: 
                    print("success")
                    data = {'certification' : 'SUCCESS'}
                    URL_recog = URL_web_reservation + str(car_id)
                    requests.put(URL_recog, json = data, headers = headers)
                    break
                else:
                    print("fail")
                    data = {'certification' : 'IDLE'}
                    URL_recog = URL_web_reservation + str(car_id)
                    requests.put(URL_recog, json = data, headers = headers)
                    break
                
        if(requests.get(URL_web_reservation, headers = headers).json()['_embedded']['reservationResourceList'][car_id-1]['certification'] != 'SUCCESS'):
            continue
        else:
            data = {'certification' : 'IDLE'}
            URL_recog = URL_web_reservation + str(car_id)
            requests.put(URL_recog, json = data, headers = headers)
            break

    # OTP
    while True:
        otp_number = requests.get(URL_recog, headers = headers).json()['otp_number']
        print(f'otp_number : {otp_number}') # check otp
        if(otp_number >= 100000):
            print(f'car id : {car_id}') # check car_id
                
            for n in range(5): # check 5 times
                otp = input("otp : ")
                if otp_number == int(otp):
                    print("success")
                    data = {'otp_number' : 1}
                    requests.put(URL_recog, json = data, headers = headers).json
                    break
                if n == 4:
                    print("fail")
                    data = {'otp_number' : 2}
                    requests.put(URL_recog, json = data, headers = headers).json
            break    

        if (requests.get(URL_recog, headers = headers).json()['otp_number'] != 1):
            continue
        else: # otp reset
            data = {'otp_number' : '2'}
            requests.put(URL_recog, json = data, headers = headers)
            break
            
    # sleep detection
    while True:
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []

            for n in range(36,42):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                leftEye.append((x,y))
                next_point = n+1
                if n == 41:
                    next_point = 36
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            for n in range(42,48):
                x = face_landmarks.part(n).x
                y = face_landmarks.part(n).y
                rightEye.append((x,y))
                next_point = n+1
                if n == 47:
                    next_point = 42
                x2 = face_landmarks.part(next_point).x
                y2 = face_landmarks.part(next_point).y
                cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)

            if EAR<0.18:
                close()
                print(f'close count : {close.count}')
                if close.count == 15:
                    print("Driver is sleeping")
                    sound()
            print(EAR)

        cv2.imshow("Are you Sleepy", frame)

        key = cv2.waitKey(30)
        if key == 27:
            break

    camera.release()


def cam2():
  camera2 = cv2.VideoCapture(1)
  front_num = 0
  URL_car = URL_web + 'api/car/456가1234' #123호1234
  
  # face_detection
  with mp_face_detection.FaceDetection(
      min_detection_confidence=0.5) as face_detection:
    while camera2.isOpened():
      success, frame2 = camera2.read()
      if not success:
        print("Ignoring empty camera frame.")
        continue

      try :
        data = client_socket.recv(1024)
        if data.decode(): # Start when receive data
              cv2.imwrite("detect.png", frame2)
              image = cv2.imread("detect.png")
              image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
              
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
              else:
                  front_num = 0
            
              # After receiving the data, send it to the server.
              if data.decode() is not None:
                print('Received from', data.decode())
                back_num = data.decode()    
                people_num = int(front_num) + int(back_num) 
                print(f"front_num : {front_num}, back_num : {back_num}, total : {people_num}")
                db_data = {'cur_people' : people_num}
                requests.put(URL_car, json = db_data, headers = headers)
              
              data = 'NULL'
          
      except :
              if cv2.waitKey(30) & 0xFF == 27:
                break
    camera2.release() 

# threads by sensor
threading.Thread(target=cam1).start()
threading.Thread(target=cam2).start()
