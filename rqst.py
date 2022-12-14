from unittest import result
import cv2
import json
import base64
import urllib3
import requests
import numpy as np
import logging



def _encode(self, img: np.ndarray) -> str:
    img = cv2.resize(img, (224, 224))  # if product101(fridge & classification)
    img = img[:, :, ::-1]  # bgr to rgb
    img = cv2.imencode('.jpg', img)[1].tobytes()  # .png or .jpg
    img = base64.b64encode(img).decode('utf-8')
    return img

# cropped_batch_images = self._get_images_and_box()
# images = [self._encode(img) for img in cropped_batch_images]

payload = {'model_path' : './weight',
                    'device_type' : 'fr',
                    'model_type' : 'cls',
                    'network_name' : 'mobilenetv3_small_075',  # mobilenetv3_small_075 고정
                    'model_name' : 'paulaner_munchner_hell_500_can',
                    'version' : '1'}


port = 8456
ip = "192.168.0.94"
api_URL = f"http://{ip}:{port}/qunt"

resp = requests.post(
    url=api_URL,
    data=json.dumps(payload),
    verify=False)

print(resp.text)


# payload = {
#     'images': 여기,
#     'params': {},``
# }
# resp = requests.post(
#     url=돌리신 서버 주소:port/endpoint(fr-cls-cass_fresh...),
#     data=json.dumps(payload),
#     headers={
#         "USER-ID": info['user_id']
#     },
#     verify=False
# )
# result = resp.json()
# print(result)