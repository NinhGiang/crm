"""
Run a rest API exposing the yolov5s object detection model
"""
import argparse
from importlib.resources import path
import io
from urllib.request import urlopen
from PIL import Image
import os
import requests
import pyrebase
import cv2
import multiprocessing
from datetime import datetime
import numpy as np

import torch

import fastai.vision.learner

import fastai.vision.data
import torchvision

from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, send
app = Flask(__name__)
CORS(app)
DETECTION_URL = "/v1/object-detection/yolov5s"
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route(DETECTION_URL, methods=["POST"])
def predict():
    print("Alo alo ko sao het")
    if not request.method == "POST":
        return None
    form = request.get_json()
    print(form)
    if form['usermediaID']:
        mediaID = form['usermediaID']
        usermedia = get_media_info(mediaID)
        albumID = usermedia['albumId']
        response = requests.get(usermedia["mediaURL"])
        print(usermedia["mediaURL"])
        if usermedia['isImage'] == True:
            img = Image.open(io.BytesIO(response.content))
            print(img.format)
            print(img.format_description)

            ### statuscode: 1 - image received
            update_media_status(usermedia, 1)
            ###

            #req = urlopen(usermedia["mediaURL"])
            #arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
            #img_temp = cv2.imdecode(arr, -1) # 'Load it as it is'
            #print(type(img_temp))
            results = model(img)

            ### statuscode: 2 - image detected, classifying
            update_media_status(usermedia, 2)
            ###

            print("da apply result")
            crop_count = 0
            pred_list = []
            crop_result_list = []
            for index, row in results.pandas().xyxy[0].iterrows(): # xet tung crop image
                crop = img.crop((row['xmin'], row['ymin'], row['xmax'], row['ymax'])) # get crop
                file_name = str(albumID) + "_" + str(mediaID) + "_" + str(crop_count) # tao file name tren firebase
                path_local = str(os.getcwd()) + "\\temp\\" + file_name + ".jpg" # tao file name local
                crop.save(path_local, format="JPEG") # save local
                print("Path ne:", path_local)
                crop_url = upload_to_firebase("crop", file_name, path_local) # upload len firebase, lay url
                r = []
                # predict resnet
                prediction_list = modelc.predict_with_mc_dropout(fastai.vision.image.open_image(path_local))
                #prediction_list = modelc.predict_with_mc_dropout(fastai.vision.image.open_image("temp/img" + ".jpg"))
                for item in prediction_list:
                    print(item[0])
                    r.append(str(item[0]))
                most, count = most_common(r)
                print(str(most) + str(count))
                # add to pred_list, tao json
                #if len(set(r)) > 1:
                if count < 5:
                    pred_list.append("Undefined")
                    crop_result_list.append(create_json_image(mediaID, crop_url, "Undefined", 1))
                else:
                    #p = set(r).pop()
                    #pred_list.append(p)
                    pred_list.append(most)
                    crop_result_list.append(create_json_image(mediaID, crop_url, most, 2))
                results.pred[0][index, 5] = torch.Tensor([index])
                crop_count += 1

            ### statuscode: 3 - image classified, generating result
            update_media_status(usermedia, 3)
            ###

            results.names = pred_list
            annotation = create_annotation(mediaID, 1, results.pandas().xyxy[0].to_json(orient="records"))
            results.render()  # updates results.imgs with boxes and labels
            path_render_img = str(os.getcwd()) + "/static/" + str(albumID) + "_" + str(mediaID) + ".jpg"
            for img in results.imgs:
                img_base64 = Image.fromarray(img)
                img_base64.save(path_render_img, format="JPEG")
            img_url = upload_to_firebase("usermedia", str(albumID) + "_" + str(mediaID) + "_detected", path_render_img) # upload len firebase, lay url
            usermedia["isDetected"] = True
            usermedia["detectedMediaURL"] = img_url
            put_to_database(usermedia, crop_result_list, annotation)

            if crop_count == 0:
                ### statuscode: 7 - done, no coral detected
                update_media_status(usermedia, 7)
                ###
            else:
                ### statuscode: 8 - done, coral detected
                update_media_status(usermedia, 8)
                ###

        else:
            print("URL la", usermedia["mediaURL"])
            generate_video(usermedia["mediaURL"], usermedia, albumID, mediaID)
    return ""

def generate_video(filepath, usermedia, albumID, mediaID):
    vid = cv2.VideoCapture(filepath) # create video object
    fps = vid.get(cv2.CAP_PROP_FPS) # get video fps
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)) # get width
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)) # get height
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT)) # get length

    ### statuscode: 4 - video received, extracting frames
    update_media_status(usermedia, 4)
    ###

    print("Num of frames: ", length)
    size = (width, height)
    video_local_path = str(os.getcwd())+ "/static/" + str(albumID) + "_" + str(mediaID) + ".mp4"
    out = cv2.VideoWriter(video_local_path,cv2.VideoWriter_fourcc(*'avc1'), fps, size) # create output video object
    results_list = []
    frames_list = get_all_frames(vid)

    ### statuscode: 5 - frames extracted, detecting and classifying
    update_media_status(usermedia, 5)
    ###

    process_video(frames_list, results_list, albumID, mediaID)

    ### statuscode: 6 - detected and classified, generating video
    update_media_status(usermedia, 6)
    ###

    results_list.sort()
    for image_path in results_list:
        out.write(cv2.imread(image_path))
    print()
    out.release()

    video_url = upload_to_firebase("usermedia", str(albumID) + "_" + str(mediaID) + "_detected", video_local_path) # upload len firebase, lay url
    usermedia["isDetected"] = True
    usermedia["detectedMediaURL"] = video_url
    put_to_database(usermedia, [], [])

    ### statuscode: 8 - done, coral detected
    update_media_status(usermedia, 8)
    ###

    #return redirect("static/video.mp4")
    return

def get_all_frames(vid):
    frames_list = multiprocessing.Queue()
    while(vid.isOpened()):
        success, frame = vid.read() # get the next frame
        if not success: # if there is no frame left, exit while loop
            break
        current_frame = int(vid.get(cv2.CAP_PROP_POS_FRAMES))
        frames_list.put([frame, current_frame])
    return frames_list

def process_video(frames_list, results_list, albumID, mediaID): # process whole video
    while not frames_list.empty():
        frame_el = frames_list.get()
        frame = frame_el[0]
        current_frame = frame_el[1]
        print("-" + str(current_frame), end = " ")
        results = model(frame, size=640) # get result from each frame
        pred_list = [] # empty list of result labels
        crop_count = 0
        for index, row in results.pandas().xyxy[0].iterrows():
            crop = Image.fromarray(frame).crop((row['xmin'], row['ymin'], row['xmax'], row['ymax'])) # get crop
            file_name = str(albumID) + "_" + str(mediaID) + "_" + str(current_frame) + "_" + str(crop_count) # tao file name tren firebase
            path_local = str(os.getcwd()) + "\\temp\\" + file_name + ".jpg"
            crop.save(path_local, format="JPEG") # save local
            #crop_url = upload_to_firebase("crop", file_name, path_local) # upload len firebase, lay url
            r = []
            prediction_list = modelc.predict_with_mc_dropout(fastai.vision.image.open_image(path_local))
            #prediction_list = modelc.predict_with_mc_dropout(fastai.vision.image.open_image("temp/img" + ".jpg"))
            for item in prediction_list:
                print(item[0])
                r.append(str(item[0]))
            most, count = most_common(r)
            #if len(set(r)) > 1:
            if count < 5:
                pred_list.append("Undefined")
                #crop_result_list.append(create_json_image(mediaID, crop_url, "Undefined"))
            else:
                #p = set(r).pop()
                #pred_list.append(p)
                pred_list.append(most)
                #crop_result_list.append(create_json_image(mediaID, crop_url, p))
            results.pred[0][index, 5] = torch.Tensor([index])
            crop_count += 1
        results.names = pred_list
        results.render()  # updates results.imgs with boxes and labels
        path_render_img = str(os.getcwd()) + "/static/" + str(albumID) + "_" + str(mediaID) + "_" + str(current_frame).rjust(8, '0') + ".jpg"
        for img in results.imgs:
            img_base64 = Image.fromarray(img)
            img_base64.save(path_render_img, format="JPEG")
        results_list.append(path_render_img)
    return

def upload_to_firebase(path_on_cloud, file_name, path_local):
    storage.child(path_on_cloud + "/" + file_name).put(path_local)
    url = storage.child(path_on_cloud + "/" + file_name).get_url(None)
    return url

def get_media_info(usermediaID):
    get_usermedia = "https://coraldetectionmodel.azurewebsites.net/api/1/UserMedia/" + str(usermediaID)
    # get usermedia info
    response = requests.get(get_usermedia).json()
    return response

def put_to_database(media_result, crop_result_list, annotation_result):
    #now = str(datetime.now().isoformat())
    # put to usermedia
    response = requests.put(put_usermedia, json=media_result)
    #print(response.content)
    print("Put to usermedia status code: ", response.status_code)
    # post to image
    i = 0
    for item in crop_result_list:
        i += 1
        response = requests.post(post_image, json=item)
        emit('updatedFile', {'data': post_image.userMediaId}, broadcast=True)
        #print(response.content)
        print("Post item " + str(i) + " to image status code: ", response.status_code)
    # post to annotation
    response = requests.post(post_annotation, json=annotation_result)
    #print(response.content)
    print("Post item to annotation status code: ", response.status_code)
    return

def create_json_image(usermediaID, imageURL, label, status):
#    return {
#        "imageId": 1,
#        "userMediaId": int(usermediaID),
#        "imageURL": imageURL,
#        "label": label,
#        "status": status,
#        "labeledBy": 1,
#        "verifiedBy": 1,
#        "isDeleted": False
#        }
    return {
    "imageId": 1,
    "userMediaId": int(usermediaID),
    "imageURL": imageURL,
    "aiLabel": label,
    "createdTime": str(datetime.now().isoformat()),
    "isDeleted": False
    }

def create_annotation(usermediaID, timestamp, jsonstring):
    return {
        "annotationId": 1,
        "userMediaId": int(usermediaID),
        "timestamp": timestamp,
        "annotationURL": jsonstring
        }

def update_media_status(usermedia, statuscode):
    temp = {
        "userMediaId": usermedia['userMediaId'],
        "albumId": usermedia['albumId'],
        "mediaURL": usermedia['mediaURL'],
        "isImage": usermedia['isImage'],
        "isDetected": usermedia['isDetected'],
        "detectedMediaURL": usermedia['detectedMediaURL'],
        "userMediaName": usermedia['userMediaName'],
        "status": statuscode,
        "createdTime": usermedia['createdTime'],
        "isDeleted": usermedia['isDeleted']
    }
    response = requests.put(put_usermedia, json=temp)
    print("Update usermedia status status code: ", response.status_code)

def clear_directories():
    dir1 = 'temp'
    for f in os.listdir(dir1):
        os.remove(os.path.join(dir1, f))
    dir2 = 'static'
    for f in os.listdir(dir2):
        os.remove(os.path.join(dir2, f))

def most_common(lst):
    a = max(set(lst), key=lst.count)
    return a, lst.count(a)

# SocketIO Events
@socketio.on('connect')
def connected():
    print('Connected')

@socketio.on('disconnect')
def disconnected():
    print('Disconnected')

@socketio.on('updatedFile')
def userAdded(file):
    print('update_media_status')

if __name__ == "__main__":
    put_usermedia = "https://coraldetectionmodel.azurewebsites.net/api/1/UserMedia"
    post_annotation = "https://coraldetectionmodel.azurewebsites.net/api/1/Annotation"
    post_image = "https://coraldetectionmodel.azurewebsites.net/api/1/Image"
    socketio.run(app, debug=True)
    parser = argparse.ArgumentParser(description="Flask api exposing yolov5 model")
    parser.add_argument("--port", default=5000, type=int, help="port number")
    args = parser.parse_args()

    clear_directories()

    model = torch.hub.load(
        "ultralytics/yolov5", "custom", path=str(os.getcwd())+'/myyolov5.pt', force_reload=True, source='github', autoshape=True
    )  # force_reload = recache latest code
    model.eval()
    data = fastai.vision.data.ImageDataBunch.from_folder(str(os.getcwd())+"/data",
                                                         size=224, num_workers=6).normalize(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    #data = ImageDataLoaders.from_folder(str(os.getcwd())+"\\data\\",
    #                                                     size=224, num_workers=6).normalize(([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    #data = ImageDataLoaders.from_folder(str(os.getcwd())+"\\data\\", size=224, num_workers=6)
    #print(type(data))
    modelc = fastai.vision.learner.cnn_learner(data, torchvision.models.resnet101, path="gs://crm-storage-v1.appspot.com/resnet/models/best_resnet101_cpu_v5.pth")
    #modelc.load("best_resnet101_cpu")


    # Your credentials after create a app web project.
    config = {
        "apiKey": "AIzaSyA2h0J3xkMGM13SU6eomUA3wJJHh8yga_o",
        "authDomain": "crm-storage-v1.firebaseapp.com",
        "projectId": "crm-storage-v1",
        "databaseURL": "https://crm-storage-v1.firebaseio.com",
        "storageBucket": "crm-storage-v1.appspot.com",
        "messagingSenderId": "335115019834",
        "appId": "1:335115019834:web:42d1c6b91a20fb8d48f703",
        "measurementId": "G-YL1H72H022"
        }
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()

    app.run(host="0.0.0.0", port=args.port, debug=True)  # debug=True causes Restarting with stat
