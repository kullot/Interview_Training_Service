# from re import template
import pandas as pd
import argparse
import requests
import json
from flask import Flask, render_template, jsonify, request, redirect, url_for, Response
from qbot import run_qbot
from werkzeug.utils import secure_filename
from time import time
import numpy as np
import random
from emo_cam import main, make_scaled_result
import io
import soundfile
from distutils.log import debug
from mfcc import read_wav, model_predict
import os
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Cairo')

import matplotlib.pyplot as plt
# plt.switch_backend('agg')
from time import sleep
plt.rcParams['font.size'] = 15.0
plt.rcParams['font.family'] = 'Malgun Gothic'
# 그래프에서 마이너스 폰트 깨지는 문제에 대한 대처
plt.rcParams['axes.unicode_minus'] = False
# from models import db

labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
wedgeprops={'width': 0.8, 'edgecolor': 'w', 'linewidth': 2}
app = Flask(__name__)
UPLOAD_FOLDER = "./static/sound/record/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route('/')
def index():
    if os.path.exists('./db/wav2emo.csv') == True :
        os.remove('./db/wav2emo.csv')
    return render_template('index.html')


@app.route('/resume', methods=["POST"])
def resume():
    q1 = request.form["nm"]
    q2 = request.form["nb"]
    df = pd.DataFrame([{'이름':q1,'문항수':int(q2)}])
    print(q1, q2)
    df.to_csv('./db/name.csv',index=False, encoding='utf-8-sig')
    q2 = range(int(q2))
    return render_template('resume.html', q1=q1, q2=q2)

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/interview')
def interview():
    df_nb = pd.read_csv('./db/name.csv')
    df_nb = df_nb['문항수']
    df_nb = int(df_nb)
    # df_nb = range(1,int(df_nb)+1)
    qna_list = pd.read_csv('./db/qna_list.csv')
    qna_list = qna_list['q_list'][0].replace('[','').replace(']','').split(',')
    print(qna_list)
    qnb_list=[]
    for idx in range(df_nb):
        print(idx)
        qna_list[idx]
        qnb_list.append([idx+1,qna_list[idx]])
        print(idx)
    print(qnb_list)
    df_nb = range(1,df_nb+1)
    return render_template('interview.html', df_nb=df_nb,qna_list=qna_list, qnb_list=qnb_list)

@app.route('/result')
def result():
    df_nb = pd.read_csv('./db/name.csv')
    df_nb = df_nb['문항수']
    df_nb = int(df_nb)
    df_nb = range(1,df_nb+1)
    df_name = pd.read_csv('./db/name.csv')
    df_wc = pd.read_csv('./db/wav2emo.csv')
    image_file = df_wc['image']
    text_emo = df_wc['text_emo']
    # df_plotly2 = dict(df_wc['text_plot'])[0]
    # df_plotly3 = dict(df_wc['mfcc_plot'])[0]
    # df_plotly1 = df_wc['text_plot']
    # df_plotly2 = df_wc['mfcc_plot']
    mfcc_emo = df_wc['mfcc_emo']
    name = df_name['이름'][0]
    scaled_result = make_scaled_result()
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # fig.savefig('./static/img/face_emo.png')

    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

    x = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    scaled_result = np.array(scaled_result)
    colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
    explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    porcent = 100.*scaled_result/scaled_result.sum()

    patches, texts = plt.pie(scaled_result, colors=colors, startangle=90, radius=1.2, explode=explode)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, scaled_result),
                                            key=lambda x: x[2],
                                            reverse=True))

    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
            fontsize=8)
    ax.set_title("표정 감정 분석")
    plt.savefig('./static/img/face_emo.png')


    # pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # plt.savefig('./static/img/face_emo.png')
    return render_template('result.html', image_file=image_file, emo = text_emo, name=name, df_nb=df_nb, mfcc_emo=mfcc_emo)

@app.route('/<id>')
def others(id):
    return id + '라는 페이지는 없습니다.'

@app.route('/qbot', methods=['POST'])
def qbot():
    msg_get = ['msg1','msg2','msg3']
    msg_list = []
    for msg in msg_get:
        msg = request.form.get(msg)
        if msg == None:
            pass
        else:
            msg_list.append(msg)
    print(msg_list, len(msg_list))
    res = {
    'code'  : 1,  # 1:정상, 0:오류 <- 설정
    'answer':run_qbot( msg_list )   #'Re>' + msg
    }
    qna = res["answer"]
    print(qna,len(qna))

    list = []
    ran_num = random.randint(0,len(qna)-1)

    for i in range(len(msg_list)):
        while ran_num in list:
            ran_num = random.randint(0,len(qna)-1)
        list.append(ran_num)
    list.sort()
    print (list)
    q_list = []
    for i in list:
        print(i,qna[i])
        q_list.append(qna[i].replace('?','').replace(',','$'))
    # encoded_keyword = urllib.parse.quote(q_list)
    # print(encoded_keyword)
    print(q_list)
    print('====================================')
    qna_df = pd.DataFrame([{'q_list':q_list}])
    qna_df.to_csv('./db/qna_list.csv',encoding='utf-8-sig',index=False)
    url = f'http://192.168.0.14:66/aisound/{q_list}'
    res = requests.get(url)
    res = res.text
    print(res)
    return jsonify( qna )

# /emo_cam 구현
@app.route('/emo_cam')
def emo_cam():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', action='store', default='sad.png',
                        help='path of image to predict')
    parser.add_argument('--model_path', action='store', default='./emo_cam/model.pth',
                        help='path of model')

    parser.add_argument('--model', action='store',
                        default='emotionnet', help='network architecture')
    
    # cnn, resnet, resmotionnet, vgg19, vgg22: 48 | vgg24: 96 | efficientnet: 224, any
    parser.add_argument('--image_size', action='store', type=int,
                        default=48, help='input image size of the network')

    # 3 for efficientnet, 1 for the rest
    parser.add_argument('--image_channel', action='store', type=int,
                        default=1, help='input image layers')

    parser.add_argument('--gpu', action='store_true', default=False,
                        help='set a switch to use GPU')
    parser.add_argument('--detect_face', action='store_true',
                        default=False, help='turn on face detection')
    args = parser.parse_args()
    return Response(main(args), mimetype='multipart/x-mixed-replace; boundary=image')

# wav로 저장
@app.route('/wav2emo', methods=['POST'])
def wav2emo():
    if 'data' in request.files:
        file = request.files['data']
        
        # Write the data to a file.
        filename = secure_filename(file.filename)
        # filename = f'qusound{count}'
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(f'{filepath}.wav')
    
        # Jump back to the beginning of the file.
        file.seek(0)
        
        # Read the audio data again.
        data, samplerate = soundfile.read(file)
        with io.BytesIO() as fio:
            soundfile.write(
                fio, 
                data, 
                samplerate=samplerate, 
                subtype='PCM_16', 
                format='wav'
            )
            data = fio.getvalue()
    # url = f'http://127.0.0.1:88/sent_wc/{filename}'
    # url = f'http://127.0.0.1:3000/sent_wc/record3'
    url = f'http://127.0.0.1:88/sent_wc/{filename}'
    res = requests.get(url).json()
    print(res)
    mcff_npy = read_wav(filename)
    _, values = model_predict(mcff_npy)

    scaled_result = values
    labels = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
    wedgeprops={'width': 0.8, 'edgecolor': 'w', 'linewidth': 2}
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # fig.savefig(f'./static/img/mfcc{filename[-1]}.png')
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

    x = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    scaled_result = np.array(scaled_result)
    colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
    explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    porcent = 100.*scaled_result/scaled_result.sum()

    patches, texts = plt.pie(scaled_result, colors=colors, startangle=90, radius=1.2, explode=explode)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, scaled_result),
                                            key=lambda x: x[2],
                                            reverse=True))

    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
            fontsize=8)
    ax.set_title("음성 감정 분석")
    plt.savefig(f'./static/img/mfcc{filename[-1]}.png')

    # pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # plt.savefig(f'./static/img/mfcc{filename[-1]}.png')

    scaled_result = res['text_plot']
    fig, ax = plt.subplots(figsize=(10, 5), subplot_kw=dict(aspect="equal"))

    x = ['공포', '놀람', '분노', '슬픔', '중립', '행복', '혐오']
    scaled_result = np.array(scaled_result)
    colors = ['#A7CECB', '#B892FF', '#E6C79C', '#F199A9', '#CCE2A3', '#FFC2E2', '#CACC90']
    explode = [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    porcent = 100.*scaled_result/scaled_result.sum()

    patches, texts = plt.pie(scaled_result, colors=colors, startangle=90, radius=1.2, explode=explode)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(x, porcent)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy =  zip(*sorted(zip(patches, labels, scaled_result),
                                            key=lambda x: x[2],
                                            reverse=True))

    plt.legend(patches, labels, loc='best', bbox_to_anchor=(-0.1, 1.),
            fontsize=8)
    ax.set_title("텍스트 감정 분석")
    plt.savefig(f'./static/img/text_emo{filename[-1]}.png')

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # fig.savefig(f'./static/img/text_emo{filename[-1]}.png')
    # pie(scaled_result, labels=labels, autopct='%.0f%%', startangle=260, counterclock=False, colors=colors, wedgeprops=wedgeprops)
    # plt.savefig(f'./static/img/text_emo{filename[-1]}.png')
    df1 = pd.DataFrame([res])
    df1['mfcc_emo'] = result
    df1['mfcc_path'] = f'./static/img/mfcc{filename[-1]}.png'
    if os.path.exists('./db/wav2emo.csv') == False :
        df1.to_csv('./db/wav2emo.csv', encoding='utf-8-sig', index=False)
    else :
        df1.to_csv('./db/wav2emo.csv', encoding='utf-8-sig', index=False, header=None, mode='a')
    print('wav2emo.csv 저장완료')
    # print(res, type(res))
    # print(res['message'], type(res['message']))
    # print(res['image'], type(res['image']))
    return

if __name__ == '__main__':
    app.run(port = 8000, debug=True, threaded=True)
    # app.run(port = 8000, debug=True)
