import librosa
import librosa.display
import numpy as np
from matplotlib import cm
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
import os
from keras import backend as K

E_emo = ['Angry', 'Anxious', 'Embarrassed', 'Happy', 'Hurt', 'Neutrality', 'Sad']
# K_emo = ['분노', '불안', '당황', '기쁨', '상처', '중립', '슬픔']
K_emo = ['분노', '공포', '놀람', '행복', '혐오', '중립', '슬픔']
N_emo = [0, 1, 2, 3, 4, 5, 6, ]

def read_wav(file_name):
    # file_path = r'./static/sound/record/'
    file_path = r'./record/'
    s_arr = np.zeros((1,13,216))
    # X, sample_rate = librosa.load(f'{file_path}{file_name}.wav', res_type='kaiser_fast', duration=2.5,sr=44000,offset=0.5)
    X, sample_rate = librosa.load(f'{file_path}{file_name}.wav', res_type='kaiser_fast', duration=2.5,sr=22050*2,offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13)
    result = np.zeros((13,216))
    result[:mfccs.shape[0],:mfccs.shape[1]] = mfccs
    s_arr[0] = result
    s_arr = np.expand_dims(s_arr,axis=3)
    print('MFCC 변환 완료!')
    return s_arr

def model_predict(s_arr):
    # model = load_model('./mfcc/best_model_11791.h5', custom_objects={"K": K})
    model = load_model('./mfcc/best_model_4442.h5', custom_objects={"K": K})
    y_prob = model.predict(s_arr, verbose=0) 
    scaled_result = np.round(y_prob, 2).flatten()*100
    predicted = y_prob.argmax(axis=-1)
    result = K_emo[int(predicted)]
    return result, scaled_result




