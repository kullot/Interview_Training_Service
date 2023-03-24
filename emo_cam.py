#!/usr/bin/env python
# coding: utf-8
from PIL import Image, ImageFont, ImageDraw
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
# from models import *
from emo_cam.models import *
import time
import pandas as pd
# import mpld3
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


#class_labels = ['happy', 'suprise', 'angry', 'anxious', 'hurt', 'sad', 'neutral']
# class_labels = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
# class_labels_dict = {'기쁨': 0, '당황': 1, '분노': 2,
#                      '불안': 3, '상처': 4, '슬픔': 5, '중립': 6}
class_labels = ['행복', '놀람', '분노', '공포', '혐오', '슬픔', '중립']
class_labels_dict = {'행복': 0, '놀람': 1, '분노': 2, '공포': 3, '혐오': 4, '슬픔': 5, '중립': 6}

face_classifier = cv2.CascadeClassifier('./emo_cam/face_classifier.xml')

display_color = (246, 189, 86)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def main(args):
    temp = []
    if args.gpu and torch.cuda.is_available():
        device = 'cuda'
        #print('GPU On')
    else:
        device = 'cpu'
        #print('GPU Off')

    model_state = torch.load(args.model_path, map_location=torch.device(device))
    model = getModel(args.model)
    model.load_state_dict(model_state['model'])

    cap = cv2.VideoCapture(0)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)   
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    # fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 30
    delay = round(1000/fps)
    # out = cv2.VideoWriter('video.avi', fourcc, fps, (int(width), int(height)))
    out = cv2.VideoWriter('./static/video/video.mp4', fourcc, fps, (int(width), int(height)))
    time.sleep(0.2)
    lastTime = time.time()*1000.0
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), display_color, 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = tt.functional.to_pil_image(roi_gray)
                roi = tt.functional.to_grayscale(roi)
                roi = tt.ToTensor()(roi).unsqueeze(0)

                # make a prediction on the ROI
                tensor = model(roi)
                probs = torch.exp(tensor).detach().numpy()
                prob = np.max(probs) * 100
                pred = torch.max(tensor, dim=1)[1].tolist()
                label = ('{}'.format(class_labels[pred[0]], prob))
                temp.append(label)

                label_position = (x, y)

                SUPPORT_UTF8 = True
                if SUPPORT_UTF8:
                    font_path = "./emo_cam/fonts/NotoSansKR-Regular.otf"
                    font = ImageFont.truetype(font_path, 32)
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    draw.text(label_position, label, font=font, fill=display_color)
                    frame = np.array(img_pil)
                else:
                    cv2.putText(frame, label, label_position,
                    cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
            else:
                cv2.putText(frame, 'No Face Found', (20, 60),
                            cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        
        # cv2.imshow('Facial Emotion Recognition', frame)
        df = pd.DataFrame(temp)
        # df.columns = ['감정']
        df.to_csv('./db/cam_emo.csv', encoding='utf-8-sig', index=False)
        # df.to_csv('./cam_emo.csv', encoding='utf-8-sig', index=False)
        out.write(frame)
        if cv2.waitKey(delay) & 0xFF == ord('q'):
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        image = buffer.tobytes()
        yield (b'--image\r\n'
        b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
    cap.release()
    out.realease()
    cv2.destroyAllWindows()
    
def make_scaled_result():
    df = pd.read_csv('./db/cam_emo.csv')
    dict_cam = {
        '공포' : [0],
        '놀람' : [0],
        '분노' : [0],
        '슬픔' : [0],
        '중립' : [0],
        '행복' : [0],
        '혐오' : [0]
    }
    df_cam = pd.DataFrame(dict_cam).T
    # display(df_zero)
    for emo, counts in zip(df.value_counts().index, df.value_counts().values):
        emo = list(emo)[0]
        if emo == "공포":
            df_cam.loc['공포'] = df_cam.loc['공포'] + counts
        elif emo == "놀람":
            df_cam.loc['놀람'] = df_cam.loc['놀람'] + counts
        elif emo == "분노":
            df_cam.loc['분노'] = df_cam.loc['분노'] + counts
        elif emo == "슬픔":
            df_cam.loc['슬픔'] = df_cam.loc['슬픔'] + counts
        elif emo == "중립":
            df_cam.loc['중립'] = df_cam.loc['중립'] + counts
        elif emo == "행복":
            df_cam.loc['행복'] = df_cam.loc['행복'] + counts
        elif emo == "혐오":
            df_cam.loc['혐오'] = df_cam.loc['혐오'] + counts

    scaled_result = df_cam[0]
    return scaled_result

import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 15.0
plt.rcParams['font.family'] = 'Malgun Gothic'
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
plt.rcParams['axes.unicode_minus'] = False


# def makeGraph(scaled_result):
    # labels =  ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    # values = scaled_result

    # # Use `hole` to create a donut-like pie chart
    # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4)])
    # fig.update_layout(
    #     title_text="감정 분석 결과",
    #     # Add annotations in the center of the donut pies.
    #     annotations=[dict(text='표정', x=0.5, y=0.5, font_size=30, showarrow=False)],
    #     margin=dict(l=30, r=30, t=30, b=30))
    # fig = fig.to_json()
    # # fig.show()


    # scaled_result = scaled_result
    # labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    # colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
    # wedgeprops={'width': 0.8, 'edgecolor': 'w', 'linewidth': 2}

    # plt.pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # plt.savefig('./static/img/face_emo.png')
    # plt.show()
    # return