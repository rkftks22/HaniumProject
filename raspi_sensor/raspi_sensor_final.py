# import package
import serial
import time
import sys
import threading
import mediapipe as mp
import cv2
import board
import adafruit_dht
import Adafruit_ADS1x15
import requests
import socket

# face_detection
def cam():
  start_time = time.time()  
  with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
      while cap.isOpened():
        success, frame = cap.read()
        if not success:
          print("Ignoring empty camera frame.")
          # If loading a video, use 'break' instead of 'continue'.
          continue
        try :
          # Send every 5 seconds
          if time.time() - start_time >=5:
            cv2.imwrite("capture.png", frame)
            start_time = time.time()
        
            image = cv2.imread("capture.png")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image.flags.writeable = False
            results = face_detection.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
            if results.detections:
              for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
    
              print(len(results.detections))    
            if results.detections is not None:
              num = len(results.detections)
            else :
              num = 0
            # data transmission  
            data = client_socket.send(str(num).encode())
            print('Send {}'.format(data.decode())) 
            start_time = time.time()
            
        except:
          if cv2.waitKey(30) & 0xFF == ord('q'):
            sys.exit()

# DHT11 : Temperature, Humidity / MQ135 : Air quality
def DHT11_MQ135():
    while True:
        try:
            # Print the values to the serial port
            temperature_c = dhtDevice.temperature
            temperature_c = round(temperature_c, 1)
            humidity = dhtDevice.humidity
            mq = int(adc1.read_adc(0, gain=GAIN, data_rate=128)/100)
            print(
                "Temp: {:.1f} C,  Humidity: {}%,  MQ : {}".format(
                    temperature_c, humidity, mq
                )
            )
            data2 = {'air_quality' : mq, 'temperature' : temperature_c, 'humidity' : humidity}
            response = requests.put(URL, json=data2, headers=headers)
        except RuntimeError as error:
            # Errors happen fairly often, DHT's are hard to read, just keep going
            print(error.args[0])
            time.sleep(2.0)
            continue
        except Exception as error:
            dhtDevice.exit()
            raise error
        if cv2.waitKey(10000) & 0xFF == ord('q'):
          sys.exit()

# JS
URL = 'http://119.70.16.37:9002/api/car/456가1234'
headers = {'Authorization' : 'Basic YWRtaW46YWRtaW4'}

# SOCKET
HOST = '192.168.0.28'
PORT = 9999
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST,PORT))
num = 0

# mediapipe model
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# video setting
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# gpio setting
adc1 = Adafruit_ADS1x15.ADS1115(address=0x48)
dhtDevice = adafruit_dht.DHT11(board.D4, use_pulseio=False)
GAIN = 1

# threads by sensor
threading.Thread(target=cam).start()
threading.Thread(target=DHT11_MQ135).start()

