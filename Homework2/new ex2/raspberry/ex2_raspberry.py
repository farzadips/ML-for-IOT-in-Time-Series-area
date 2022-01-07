import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow import keras
import zlib
from platform import python_version
import tensorflow_model_optimization as tfmot   
import tempfile
print(f"Python version used to excute the code is {python_version()}")
#import warnings
#warnings.filterwarnings('ignore')
from classes import read_audios
from classes import SignalGenerator
from classes import make_models
from classes import model_analysis
from classes import latency

#some initializations
version = "a"
epochs = 20
units = 8                        
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#reading the data

# zip_path = tf.keras.utils.get_file(
#     origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
#     fname='mini_speech_commands.zip',
#     extract=True,
#     cache_dir='.', cache_subdir='data')

# data_dir = os.path.join('.', 'data', 'mini_speech_commands')
data_dir = 'data/mini_speech_commands'
reading_class = read_audios(data_dir)
train_files, val_files, test_files = reading_class.read()

df_q = pd.read('quantization.csv')
df_qwa = pd.read('quantization_WA.csv')

df = df_q
#run the tests:

LABELS = np.array(['stop', 'up', 'yes', 'right', 'left', 'no',  'down', 'go'] , dtype = str) 

for i in range(df.shape[0]):
    print("configuration  " + str(i) + " from  " + str(df.shape[0])+ "!!!!!!!!!!!!")
    model_version = f"_V_{version}_alpha={df.loc[i]['alpha']}"
    mymodel = df.loc[i]['model'] + model_version
    TFLITE =  f'{mymodel}.tflite' 

    STFT_OPTIONS = {'frame_length': df['frame_length'][i], 'frame_step': df['frame_step'][i], 'mfcc': False}
    MFCC_OPTIONS = { 'sampling_rate': df['sample_rate'][i],'frame_length': df['frame_length'][i], 'frame_step': df['frame_step'][i], 'mfcc': True,
            'lower_frequency': df['lower_freq'][i], 'upper_frequency': df['upper_freq'][i], 'num_mel_bins': df['number_of_bins'][i],
            'num_coefficients': 10}
    if df.loc[i]['mfcc'] is True:
        options = MFCC_OPTIONS
        strides = [2, 1]
    else:
        options = STFT_OPTIONS
        strides = [2, 2]

    generator = SignalGenerator(LABELS, **options)
    train_ds = generator.make_dataset(train_files, True)
    val_ds = generator.make_dataset(val_files, False)
    test_ds = generator.make_dataset(test_files, False)

    model_maker = make_models()
    ############ Applying Structured-Based Pruning
    model, model_checkpoint_callback, checkpoint_filepath = model_maker.models(df['alpha'][i], strides, units, model_version, df['mfcc'][i], mymodel,False,train_ds)
    ############ Applying Magnitude-Based Pruning
    #model, model_checkpoint_callback, checkpoint_filepath = model_maker.models(1, strides, units, model_version, mfcc, mymodel)
    history = model.fit(train_ds, epochs=epochs,   validation_data=val_ds,callbacks=[model_checkpoint_callback ], verbose=0)
    model_maker.plot_loss(history, mymodel)
    analysis = model_analysis(test_ds, checkpoint_filepath, train_ds)
    Compressed , tflite_model_dir = analysis.S_pruning_Model_evaluate_and_compress_to_TFlite( tflite_model_dir = TFLITE)
    acc, size = analysis.load_and_evaluation(tflite_model_dir, Compressed)
    laten = latency()
    inf, tot = laten.calculate(model = tflite_model_dir, mfcc = df['mfcc'][i] ,rate = df['sample_rate'][i], length = df['frame_length'][i], stride = df['frame_step'][i],lower_frequency = df['lower_freq'][i], upper_frequency = df['upper_freq'][i], num_mel_bins = df['number_of_bins'][i])
    df['n_acuracy'][i] = acc
    df['n_size'][i] = size
    df['n_latency'][i] = tot
    print("normal done  $$$$$$$$$$$$$$$$$$$$$$$")
    #quantized:
    Compressed , Quantized   = analysis.apply_Quantization(TFLITE, PQT=True , WAPQT = False)
    acc, size = analysis.load_and_evaluation(Quantized , Compressed)
    inf, tot = laten.calculate(model = Quantized, mfcc = df['mfcc'][i] ,rate = df['sample_rate'][i], length = df['frame_length'][i], stride = df['frame_step'][i],lower_frequency = df['lower_freq'][i], upper_frequency = df['upper_freq'][i], num_mel_bins = df['number_of_bins'][i])
    df['q_acuracy'][i] = acc
    df['q_size'][i] = size
    df['q_latency'][i] = tot
    print("quantized done  $$$$$$$$$$$$$$$$$$$$$$$")
    #WA_quantized:
    WA_Compressed , WA_Quantized   = analysis.apply_Quantization(TFLITE, PQT=False ,WAPQT=True)
    acc, size = analysis.load_and_evaluation(WA_Quantized , WA_Compressed)
    inf, tot = laten.calculate(model = Quantized, mfcc = df['mfcc'][i] ,rate = df['sample_rate'][i], length = df['frame_length'][i], stride = df['frame_step'][i],lower_frequency = df['lower_freq'][i], upper_frequency = df['upper_freq'][i], num_mel_bins = df['number_of_bins'][i])
    df['qwa_acuracy'][i] = acc
    df['qwa_size'][i] = size
    df['qwa_latency'][i] = tot
    print("qWA done  $$$$$$$$$$$$$$$$$$$$$$$")
    #magnitute:
    model, model_checkpoint_callback, checkpoint_filepath = model_maker.models(df['alpha'][i], strides, units, model_version, df['mfcc'][i], mymodel,True,train_ds)
    history = model.fit(train_ds, epochs=epochs,   validation_data=val_ds,callbacks=[model_checkpoint_callback ], verbose=0)
    analysis = model_analysis(test_ds, checkpoint_filepath, train_ds)
    Compressed , tflite_model_dir = analysis.S_pruning_Model_evaluate_and_compress_to_TFlite( tflite_model_dir = TFLITE)
    acc, size = analysis.load_and_evaluation(tflite_model_dir, Compressed)
    laten = latency()
    inf, tot = laten.calculate(model = tflite_model_dir, mfcc = df['mfcc'][i] ,rate = df['sample_rate'][i], length = df['frame_length'][i], stride = df['frame_step'][i],lower_frequency = df['lower_freq'][i], upper_frequency = df['upper_freq'][i], num_mel_bins = df['number_of_bins'][i])
    df['m_acuracy'][i] = acc
    df['m_size'][i] = size
    df['m_latency'][i] = tot
    print("magnitute done  $$$$$$$$$$$$$$$$$$$$$$$")


df.to_csv('quantization_results.csv')
print("all done!!!!!!!! :)")