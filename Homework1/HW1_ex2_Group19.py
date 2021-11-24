import os
import tensorflow as tf
import time
import numpy as np
import wave
from os import listdir
import argparse

from subprocess import Popen
Popen('sudo sh -c "echo performance >'
 '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
 shell=True).wait()

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
parser.add_argument('--length', type=float, help='frame length in s for STFT',
        required=True)
parser.add_argument('--stride', type=float, help='stride in s for STFT', required=True)
parser.add_argument('--sampling_rate', type=int, help='sampling rate of audio used for STFT, MFCCslow and MFCCfast', required=True)
parser.add_argument('--nb_bins',type=int,help='number of mel filters',required=True)
parser.add_argument('--lower_freq',type=int,help='lower frequency for mel filters', required=True)
parser.add_argument('--upper_freq',type=int,help='upper frequency for mel filters',required=True)
parser.add_argument('--nb_coef',type=int,help='number of coefficients to take from the last dimension of the MFCC')


args = parser.parse_args()

def STFT(window_length,stride,sampling_rate):
    spectrogram=[]
    #l=0.016
    #s=0.008

    frame_length= int(window_length*sampling_rate)
    frame_step=int(stride*sampling_rate)

    for i in range(len(tf_audio)):
        stft = tf.signal.stft(tf_audio[i],frame_length=frame_length,frame_step=frame_step,fft_length=frame_length)
        spectrogram.append(tf.abs(stft))
        #print('\r',i,end='')
    return spectrogram

def MFCC(spectrogram,num_mel_bins,lower_freq,upper_freq,sampling_rate,nb_coef):
    mfccsf=[]
    num_spectrogram_bins=spectrogram[0].shape[-1]
    linear_to_mel_weight_matrix=tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sampling_rate, lower_freq, upper_freq)
    for i in range(len(spectrogram)):
        mel_spectrogram=tf.tensordot(spectrogram[i], linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram=tf.math.log(mel_spectrogram + 1.e-6)
        mfccsf.append(tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :nb_coef])
        #print('\r',i,end='')

    return mfccsf

def SNR (MFCCs,MFCCf):
    num=np.linalg.norm(MFCCs)
    den=np.linalg.norm(MFCCs-MFCCf+1.e-6)
    SNR=float(20*np.log10(num/den))
    return SNR

start=time.time()
os.chdir(args.filename)
onlyfiles = os.listdir()


tf_audio=[]
for i in range(len(onlyfiles)):
    audio = tf.audio.decode_wav(tf.io.read_file('{0}/{1}'.format(args.filename,onlyfiles[i])))
    tf_audio.append(audio[0])
    tf_audio[i] = tf.squeeze(tf_audio[i], 1)
del onlyfiles[:]


#print('STFT starting')   
spectrogram=STFT(args.length,args.stride,args.sampling_rate)
del tf_audio[:]
#print('STFT done')
start_mfcc=time.time()
MFCC_slow=MFCC(spectrogram,40,20,4000,16000,10)
#print('MFCCslow done')
end_mfcc=time.time()
#print('MFCC slow shape:{0}'.format(MFCC_slow[0].shape))
MFCC_fast=MFCC(spectrogram,args.nb_bins,args.lower_freq,args.upper_freq,args.sampling_rate,args.nb_coef)
#print('MFCC fast shape:{0}'.format(MFCC_fast[0].shape))
end = time.time()
print('MFCC slow = {:.0f} ms'.format(1000*(end_mfcc-start)/len(MFCC_slow)))
print('MFCC fast = {:.0f} ms'.format((1000*((end-start)-(end_mfcc-start_mfcc)))/len(MFCC_slow)))

snr_matrix=[]
for i in range(len(MFCC_slow)):
    snr_matrix.append(SNR(MFCC_slow[i],MFCC_fast[i]))
print('SNR={:.2f} dB'.format(sum(snr_matrix)/len(snr_matrix)))




#our grid search code that we ran on a different script:

"""
import os
import tensorflow as tf
import time
import numpy as np
import pandas as pd
import wave
from os import listdir
import argparse


from subprocess import Popen
Popen('sudo sh -c "echo performance >'
 '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
 shell=True).wait()

parser = argparse.ArgumentParser()
parser.add_argument('--filename', type=str, help='input name',
        required=True)
#parser.add_argument('--length', type=float, help='frame length in s for STFT',required=True)
#parser.add_argument('--stride', type=float, help='stride in s for STFT', required=True)
#parser.add_argument('--sampling_rate', type=int, help='sampling rate of audio used for STFT, MFCCslow and MFCCfast', required=True)
#parser.add_argument('--nb_bins',type=int,help='number of mel filters',required=True)
#parser.add_argument('--lower_freq',type=int,help='lower frequency for mel filters', required=True)
#parser.add_argument('--upper_freq',type=int,help='upper frequency for mel filters',required=True)
#parser.add_argument('--nb_coef',type=int,help='number of coefficients to take from the last dimension of the MFCC')


args = parser.parse_args()

def STFT(window_length,stride,sampling_rate):
    spectrogram=[]
    #l=0.016
    #s=0.008

    frame_length= int(window_length*sampling_rate)
    frame_step=int(stride*sampling_rate)

    for i in range(len(tf_audio)):
        stft = tf.signal.stft(tf_audio[i],frame_length=frame_length,frame_step=frame_step,fft_length=frame_length)
        spectrogram.append(tf.abs(stft))
        #print('\r',i,end='')
    return spectrogram

def MFCC(spectrogram,num_mel_bins,lower_freq,upper_freq,sampling_rate,nb_coef):
    mfccsf=[]
    num_spectrogram_bins=spectrogram[0].shape[-1]
    linear_to_mel_weight_matrix=tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins, num_spectrogram_bins, sampling_rate, lower_freq, upper_freq)
    for i in range(len(spectrogram)):
        mel_spectrogram=tf.tensordot(spectrogram[i], linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram=tf.math.log(mel_spectrogram + 1.e-6)
        mfccsf.append(tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[..., :nb_coef])
        #print('\r',i,end='')

    return mfccsf

def SNR (MFCCs,MFCCf):
    num=np.linalg.norm(MFCCs)
    den=np.linalg.norm(MFCCs-MFCCf+1.e-6)
    SNR=float(20*np.log10(num/den))
    return SNR


number_of_bins = [20,22,24,26,28,30,32,34,36,38,40]
lower_freq = [20,30,40,50,60,70,80,90,100,110,150,200,400,1000,2000]
upper_freq = [1000,1500,1800,2000,2200,2400,2600,2800,3000,4000]
sample_rate = [8000,16000]
df = pd.DataFrame(columns=['number_of_bins', 'lower_freq', 'upper_freq','sample_rate','snr','time'])
for p1 in range(len(number_of_bins)):
    for p2 in range(len(lower_freq)):
        for p3 in range(len(upper_freq)):
            for p4 in range(len(sample_rate)):
                if((lower_freq[p2] < upper_freq[p3]) and (upper_freq[p3] <= sample_rate[p4]/2)):
                    data = [[number_of_bins[p1],lower_freq[p2],upper_freq[p3],sample_rate[p4],0,0]]
                    d = pd.DataFrame(data, columns=['number_of_bins', 'lower_freq', 'upper_freq','sample_rate','snr','time'])
                    df = df.append(d, ignore_index = True)



main_start=time.time()
os.chdir(args.filename)
onlyfiles = os.listdir()


tf_audio=[]
for i in range(len(onlyfiles)):
    audio = tf.audio.decode_wav(tf.io.read_file('{0}/{1}'.format(args.filename,onlyfiles[i])))
    tf_audio.append(audio[0])
    tf_audio[i] = tf.squeeze(tf_audio[i], 1)
del onlyfiles[:]


#print('STFT starting')   
spectrogram=STFT(0.016,0.008,16000)
del tf_audio[:]
#print('STFT done')
main_end = time.time()
MFCC_slow=MFCC(spectrogram,40,20,4000,16000,10)
print('MFCCslow done')
for i in range(df.shape[0]):
    mfcc_start = time.time()

    MFCC_fast=MFCC(spectrogram,df.loc[i][0],df.loc[i][1],df.loc[i][2],df.loc[i][3],10)
    mfcc_end = time.time()
    df.loc[i][5] = ((1000*((main_end-main_start)+(mfcc_end-mfcc_start)))/len(MFCC_slow))
    snr_matrix=[]
    for j in range(len(MFCC_slow)):
        snr_matrix.append(SNR(MFCC_slow[j],MFCC_fast[j]))
    df.loc[i][4] = (sum(snr_matrix)/len(snr_matrix))
    print('\r',str(i) + " / " + str(df.shape[0]),end='')

df = df[df['snr'] > 10.4]
#df = df.sort_values('time',ascending = False).head(10)
os.chdir('..')
df.to_csv('best.csv')
df = df.sort_values('time',ascending = False).head(10)
df.to_csv('best_10.csv')
"""

