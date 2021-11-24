import dlib, cv2
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

# 얼굴인식 모델 정의
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./face_recognition/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./face_recognition/models/dlib_face_recognition_resnet_model_v1.dat')

user_name = 0
car_id = 0
certification_name = ""
data = {}

headers = {'Authorization' : ''} 
camera = cv2.VideoCapture(0)


def img_load(url): # url로부터 img 불러오기
    image_nparray = np.asarray(bytearray(requests.get(url).content), dtype=np.uint8)
    img = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    return img

def find_faces(img): # 얼굴 검출
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)
    
    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)
        
        # dlib 랜드마크 numpy 배열로 변환
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)
        
    return rects, shapes, shapes_np

def encode_faces(img, shapes): # 얼굴 배열 형태로 변환 encode
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor)) # 튜플에 추가

    return np.array(face_descriptors)

img_paths = {}
descs = {}
url_string = '' 

URL = ''
for a in range(len(requests.get(URL, headers = headers).json()['_embedded']['userResourceList'])):
    img_url = url_string+str(a+1)+'.jpg'
    img_paths[a+1] = img_url
    descs[a+1] = None

for name, img_path in img_paths.items(): # 이미지 불러와 튜플에 저장
    img_bgr = img_load(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb) 
    descs[name] = encode_faces(img_rgb, img_shapes)[0]


np.save('./face_recognition/img/descs.npy', descs)

URL = ''
while True:
    for i in range(len(requests.get(URL, headers = headers).json()['_embedded']['reservationResourceList'])):
        print(f'num : {i}')
        recog = requests.get(URL, headers = headers).json()['_embedded']['reservationResourceList'][i]['reservationDto']['certification']
        print(f'recog : {recog}')
        if(recog == 'DOING'):
            user_name = requests.get(URL, headers = headers).json()['_embedded']['reservationResourceList'][i]['reservationDto']['userId']
            car_id = requests.get(URL, headers = headers).json()['_embedded']['reservationResourceList'][i]['reservationDto']['reservationId']
            print(f'user name : {user_name} / car id = {car_id}')
        #
            ret,frame = camera.read()
            cv2.imwrite("webcam.jpg", frame)
            print("capture")

            img_bgr = cv2.imread('./face_recognition/img/bale.jpg')
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            rects, shapes, _ = find_faces(img_rgb)
            descriptors = encode_faces(img_rgb, shapes)
        #  
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
                        print(f'name : {certification_name}')
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
                URL = '' + str(car_id)
                print(URL)
                requests.put(URL, json = data, headers = headers)
                break
            else:
                print("fail")
                data = {'certification' : 'IDLE'}
                URL = '' + str(car_id)
                print(URL)
                requests.put(URL, json = data, headers = headers)
                break
    URL = ''
    if(requests.get(URL, headers = headers).json()['_embedded']['reservationResourceList'][car_id-1]['reservationDto']['certification'] != 'SUCCESS'):
        continue
    else:
        data = {'certification' : 'IDLE'}
        URL = '' + str(car_id)
        requests.put(URL, json = data, headers = headers)
        URL = ''
        break
    
camera.release()