import streamlit as st
import soundfile as sf
import os
import shutil
import boto3
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
import parselmouth
from parselmouth.praat import call, run_file
import glob
import librosa
import errno
import csv,sys
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import os
from subprocess import check_output
import queue
import soundfile as sf
import _thread  
import pickle
from scipy.stats import binom
from scipy.stats import ks_2samp
from scipy.stats import ttest_ind
from pandas import read_csv

def get_analysis():
    pa0 = 'temp2.wav'
    files = glob.glob(pa0)
    pa00 = files[0].replace('temp2.wav',os.getcwd())
    pa8= 'MLTRNL.praat'
    result_array = np.empty((0, 100))
    result_array = np.empty((0, 27))
    objects= run_file(pa8, -20, 2, 0.3, "yes", pa0,pa00, 80, 400, 0.01, capture_output=True)
    #print (objects[0]) # This will print the info from the sound object, and objects[0] is a parselmouth.Sound object
    z1=( objects[1]) # This will print the info from the textgrid object, and objects[1] is a parselmouth.Data object with a TextGrid inside
    z3=z1.strip().split()
    z2=np.array([z3])
    result_array=np.append(result_array,[z3], axis=0)
    np.savetxt('temp.csv',result_array, fmt='%s',delimiter=',')

    #Data and features analysis 
    df = pd.read_csv('temp.csv',
                        names = ['avepauseduratin','avelongpause','speakingtot','avenumberofwords','articulationrate','inpro','f1norm','mr','q25',
                                'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','formantmean','formantstd','nuofwrds','npause','ins',
                                'fillerratio','xx','xxx','totsco','xxban','speakingrate'],na_values='?')
    #scoreMLdataset.to_csv(pa7, header=False,index = False)
    newMLdataset=df.drop(['avelongpause','speakingtot','avenumberofwords','inpro','f1norm','mr','q25',
                                'q50','q75','std','fmax','fmin','vowelinx1','vowelinx2','nuofwrds','ins',
                                'xx','xxx','totsco','xxban','speakingrate'], axis=1)
    os.remove('temp.csv')
    return newMLdataset

def feature_extraction(file_name):
    #X, sample_rate = sf.read(file_name, dtype='float32')
    X , sample_rate = librosa.load(file_name, sr=None) #Can also load file using librosa
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    
    ## stFourier Transform
    stft = np.abs(librosa.stft(X))
            
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=20).T, axis=0) #Returns N_mel coefs
    rmse = np.mean(librosa.feature.rms(y=X).T, axis=0) #RMS Energy for each Frame (Stanford's). Returns 1 value 
    spectral_flux = np.mean(librosa.onset.onset_strength(y=X, sr=sample_rate).T, axis=0) #Spectral Flux (Stanford's). Returns 1 Value
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=X).T, axis=0) #Returns 1 value
    
    #mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0) #Returns 128 values
    #chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0) #Returns 12 values
    #contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0) #Returns 7 values
    #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0) #tonal centroid features Returns 6 values
    
    ##Return computed audio features
    return mfccs, rmse, spectral_flux, zcr

def parse_single_audio(file_path): # Audio Format
    n_mfccs = 20 # This variable is tunneable with each run
    number_of_features = 3 + n_mfccs
    #number_of_features = 154 + n_mfccs # 154 are the total values returned by rest of computed features
    features, labels = np.empty((0,number_of_features)), np.empty(0)
    
    ##Extract features for each audio file
    try:
        mfccs, rmse, spectral_flux, zcr = feature_extraction(file_path)
        #mfccs, zcr, mel, chroma, contrast, tonnetz = feature_extraction(file_name)
    except Exception as e:
        print("[Error] there was an error in feature extraction. %s" % (e))      
    extracted_features = np.hstack([mfccs, rmse, spectral_flux, zcr])
    #print "Total Extracted Features: ", len(extracted_features) #This helps us identify really how many features are being computed
    features = np.vstack([features, extracted_features]) #Stack arrays in sequence vertically (row wise).
    print("Extracted features from %s, done" % (file_path))
    return features ## arrays with features and corresponding labels for each audio
    
def process_mp3(file_path):
    query = f'ffmpeg -i {file_path} -vn -ar 32000 -ac 2 -b:a 32k "temp2.wav"'
    os.system(query)

st.caption('Currently supports only .wav file')
uploaded_file = st.file_uploader("Upload an audio clip to analyse")

if uploaded_file is not None:
    # To read file as bytes:
    audio_data, audio_samplerate = sf.read(uploaded_file)
    sf.write('temp.wav', audio_data, audio_samplerate)
    #audio.export(uploaded_file,format=file_type)    
    #audio = pydub.AudioSegment.from_wav(uploaded_file)
    process_mp3('temp.wav')
    st.audio(uploaded_file.getvalue(), format='audio/wav')
    os.remove('temp.wav')
    import pickle
    svm_clf = pickle.load(open('Fluency_using_SVM','rb'))
    df = get_analysis()
    audio_as_np = parse_single_audio('temp2.wav')
    fluency = svm_clf.predict_proba(audio_as_np)
    df['fluency'] = fluency.tolist()
    st.dataframe(df)
    os.remove('temp2.wav')
    df2 = {'fluency':'','rateofspeaking':'','usage of fillers':'','pause duration':''}
    fluency = fluency.tolist()[0]
    if fluency[1] > .8:
        df2['fluency'] = 'high'
    else:
        label = np.argmax(fluency)
        if label == 0:
            df2['fluency'] = 'low'
        elif label == 1:
            df2['fluency'] = 'intermediate'
        elif label == 2:
            df2['fluency'] = 'high'       
    if df['articulationrate'][0] >=4.3 and df['articulationrate'][0] <= 4.6:
        df2['rateofspeaking'] = 'normal'
    elif df['articulationrate'][0] < 4.3:
        df2['rateofspeaking'] = 'slow'
    elif df['articulationrate'][0] > 4.6:
        df2['rateofspeaking'] = 'fast'
    if df['fillerratio'][0] >10:
        df2['usage of fillers'] = 'high'
    else:
        df2['usage of fillers'] = 'normal'    
    if df['avepauseduratin'][0] >= .55 and df['avepauseduratin'][0] <= .6:
        df2['pause duration'] = 'normal'
    elif df['avepauseduratin'][0] < .55:
        df2['pause duration'] = 'less'
    elif df['avepauseduratin'][0] > .6:
        df2['pause duration'] = 'more'    
    st.header('Categorised csv') 
    df3 = pd.DataFrame([df2])
    st.dataframe(df3)
