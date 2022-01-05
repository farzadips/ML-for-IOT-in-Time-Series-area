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
import matplotlib.pyplot as plt
import tensorflow_model_optimization as tfmot







class read_audios:
    def __init__(self, path):
        self.path  = path
    
    def read(self):
        filenames = tf.io.gfile.glob(self.path + '/*/*')
        filenames = tf.random.shuffle(filenames)
        total = 8000
        train_files = filenames[:int(total*0.8)]
        val_files = filenames[int(total*0.8): int(total*0.9)]
        test_files = filenames[int(total*0.9):]
        return train_files, val_files, test_files





class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
            num_mel_bins=None, lower_frequency=None, upper_frequency=None,
            num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate                                             # 16000  
        self.frame_length = frame_length                                               # 640 
        self.frame_step = frame_step                                                   # 320 
        self.num_mel_bins = num_mel_bins                                               # 40 
        self.lower_frequency = lower_frequency                                         # 20 
        self.upper_frequency = upper_frequency                                         # 4000
        self.num_coefficients = num_coefficients                                       # 10 
        num_spectrogram_bins = (frame_length) // 2 + 1                                  # ( frame size // 2 ) + 1 

        '''
        STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
        MFCC_OPTIONS = {'frame_length': 640, 'frame_step': 320, 'mfcc': True,
        'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 40,
        'num_coefficients': 10}
        '''

        if mfcc is True:                                                                # Remember we need to compute this matrix once so it will be a class argument 
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                    self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                    self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]                                  # -1 is audio.wav so 
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        # Padding for files with less than 16000 samples
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)     # if the shape of the audio is already = 16000 (sampling rate) we will add nothing 

        # Concatenate audio with padding so that all audio clips will be of the  same length
        audio = tf.concat([audio, zero_padding], 0)
        # Unify the shape to the sampling frequency (16000 , )
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)                         # expand_dims will not add or reduce elements in a tensor, it just changes the shape by adding 1 to dimensions for the batchs. 
    
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files, train):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls = tf.data.experimental.AUTOTUNE) # better than 4 tf.data.experimental.AUTOTUNE will use the maximum num_parallel_calls 
        ds = ds.batch(32)
        ds = ds.cache()
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


class make_models():
    def __init__(self):
        pass

    def models(self,alpha, strides, units, model_version, mfcc, mymodel):
        mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units = int(256 *alpha), activation='relu' , name =  "Dense-1" ),
            tf.keras.layers.Dense(units = int(256 *alpha), activation='relu', name =  "Dense-2"),
            tf.keras.layers.Dense(units = int(256 *alpha), activation='relu', name =   "Dense-3" ),
            tf.keras.layers.Dense(units = units , name =  "Output-Layer")                                   # change to 9 if silence included 
        ])

        cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=int(128 *alpha), kernel_size=[3,3], strides=strides, use_bias=False , name = "Conv2D-1"),
            tf.keras.layers.BatchNormalization(momentum=0.1 , name = "Btch_Norm-1"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=int(128 *alpha), kernel_size=[3,3], strides=[1,1], use_bias=False , name = "Conv2D-2"),
            tf.keras.layers.BatchNormalization(momentum=0.1 , name = "Btch_Norm-2"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(filters=int(128 *alpha), kernel_size=[3,3], strides=[1,1], use_bias=False , name = "Conv2D-3"),
            tf.keras.layers.BatchNormalization(momentum=0.1 , name = "Btch_Norm-3"),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D( name =  "GlobalAveragePooling-Layer"),
            tf.keras.layers.Dense(units = units, name =  "Output-Layer")
        ])

        ds_cnn = tf.keras.Sequential([
            tf.keras.layers.Conv2D(filters=int(256 *alpha), kernel_size=[3,3], strides=strides, use_bias=False, name = "Conv2D-1"),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, name = "DepthwiseConv2D-1"),
            tf.keras.layers.Conv2D(filters=int(256 *alpha), kernel_size=[1,1], strides=[1,1], use_bias=False, name = "Conv2D-2"),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False, name = "DepthwiseConv2D-2"),
            tf.keras.layers.Conv2D(filters=int(256 *alpha), kernel_size=[1,1], strides=[1,1], use_bias=False, name = "Conv2D-3"),
            tf.keras.layers.BatchNormalization(momentum=0.1),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling2D( name =  "GlobalAveragePooling-Layer"),
            tf.keras.layers.Dense(units = units, name =  "Output-Layer")
        ])


        MODELS = {'mlp'+ model_version : mlp, 'cnn'+ model_version: cnn, 'ds_cnn'+ model_version: ds_cnn}
        model = MODELS[mymodel] 
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        if mfcc is False:
            checkpoint_filepath = f'./checkpoints/stft/chkp_best_{mymodel}'

        else:
            checkpoint_filepath = f'./checkpoints/mfcc/chkp_best_{mymodel}'
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,           
            monitor='val_sparse_categorical_accuracy',
            verbose=1,
            mode='max',
            save_best_only=True,
            save_freq='epoch')

        return model, model_checkpoint_callback, checkpoint_filepath

    def plot_loss(slef, history, mymodel):
        plt.plot(history.history['sparse_categorical_accuracy'], label='Accuracy')
        plt.plot(history.history['val_sparse_categorical_accuracy'], label='val_Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(mymodel+".png")


    

class model_analysis():
    def __init__(self, test_ds, checkpoint_filepath, train_ds):
        self.test_ds = test_ds
        self.checkpoint_filepath = checkpoint_filepath
        self.train_ds = train_ds

    def S_pruning_Model_evaluate_and_compress_to_TFlite(self, tflite_model_dir):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        best_model = tf.keras.models.load_model(filepath = self.checkpoint_filepath )
        Loss , ACCURACY = best_model.evaluate(self.test_ds)
        print("*"*50,"\n",f" The accuracy achieved by the best model before convertion = {ACCURACY *100:0.2f}% ")
        # Convert to TF lite without Quantization 
        converter = tf.lite.TFLiteConverter.from_saved_model(self.checkpoint_filepath)
        tflite_model = converter.convert()  
        Compressed = "compressed_"+tflite_model_dir 
        tflite_model_dir = './models/'+tflite_model_dir
        # Write the model in binary formate and save it 
        with open(tflite_model_dir, 'wb') as fp:
            fp.write(tflite_model)
        Compressed = './models/'+Compressed
        with open(Compressed, 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)
        print("*"*50,"\n",f"the model is saved successfuly to {tflite_model_dir}")
        return Compressed , tflite_model_dir 

    def getsize(self, file):
        st = os.stat(file)
        size = st.st_size
        return size

    def load_and_evaluation(self, path , Compressed):
        interpreter = tf.lite.Interpreter(model_path = path) 
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dataset = self.test_ds.unbatch().batch(1)
        
        COMMANDS = ['stop', 'up', 'yes', 'right', 'left', 'no',  'down', 'go']
        
        outputs = []
        labels = []
        count = 0                                 # counter to compute the number of correct predictions 
        total = 0                                 # total number of samples / predictions ==> acc = count/total
        
        for inp , label in dataset:
            my_input = np.array(inp, dtype = np.float32)
            label = np.array(label, dtype = np.float32)
        
            
            labels.append(label)

            interpreter.set_tensor(input_details[0]['index'], my_input)
            interpreter.invoke()
            my_output = interpreter.get_tensor(output_details[0]['index'])
            predict = np.argmax(my_output)                                 # the prediction crossponds to the index of with the highest probability   
            outputs.append(predict)
            total += 1   
            if (predict == label):                                         # if probability == labesl increase the correct predictions counter 
                count += 1
        # Compute the Accuracy         
        accuracy = count/total*100
        # Evaluate the size of Tflite model 
        size = self.getsize(path)/1000
        # Evaluate the size of Tflite model  after Comperession 
        size_compressed = self.getsize(Compressed)/1000
        print ("*"*50,"\n",f"The Size of TF lite model  Before compression is = {size} kb" )
        print ("*"*50,"\n",f"The Size of TF lite model  After compression is = {size_compressed} kb" )
        print ("*"*50,"\n",f"The accuracy of TF lite model is = {accuracy:0.2f} " )
        return accuracy, size_compressed 

        # Function for weight and activations quantization 
    def representative_dataset_gen(self):
        for x, _ in self.train_ds.take(1000):
            yield [x]



    def apply_Quantization(self, tflite_model_dir, PQT = False, WAPQT = False): 

        converter = tf.lite.TFLiteConverter.from_saved_model(self.checkpoint_filepath)
        
        # Apply weight only quantization 
        if PQT == True :
            tflite_model_dir = f"PQT_{tflite_model_dir}"
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()
        # Apply weight + Activation  quantization 
        if WAPQT == True :
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.representative_dataset_gen
            tflite_model = converter.convert()
        
            tflite_model_dir = f"WAPQT_{tflite_model_dir}"
        Compressed =  f"compressed_{tflite_model_dir}"
        tflite_model_dir =   f"./models/{tflite_model_dir}"
        # Write the model in binary formate and save it 
        with open(tflite_model_dir, 'wb') as fp:
            fp.write(tflite_model)
        Compressed = f"./models/{Compressed}"
        with open(Compressed, 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)
        print(f"the model is saved successfuly to {tflite_model_dir}")
        return Compressed , tflite_model_dir 


class latency():
    def __init__(self):
        pass

    def calculate(self, model, rate = 16000, mfcc = False, resize = 32, length = 640, stride = 320, num_mel_bins = 40, lower_frequency = 20, upper_frequency = 4000, num_coefficients = 10):
        import tensorflow as tf
        import time
        from scipy import signal
        import numpy as np
        from subprocess import call


        num_frames = (rate - length) // stride + 1
        num_spectrogram_bins = length // 2 + 1

        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, rate, lower_frequency,
                upper_frequency)

        if model is not None:
            interpreter = tf.lite.Interpreter(model_path = model)
            interpreter.allocate_tensors()

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()


        inf_latency = []
        tot_latency = []
        for i in range(100):
            sample = np.array(np.random.random_sample(48000), dtype=np.float32)

            start = time.time()

            # Resampling
            sample = signal.resample_poly(sample, 1, 48000 // rate)

            sample = tf.convert_to_tensor(sample, dtype=tf.float32)

            # STFT
            stft = tf.signal.stft(sample, length, stride,
                    fft_length=length)
            spectrogram = tf.abs(stft)

            if mfcc is False and resize > 0:
                # Resize (optional)
                spectrogram = tf.reshape(spectrogram, [1, num_frames, num_spectrogram_bins, 1])
                spectrogram = tf.image.resize(spectrogram, [resize, resize])
                input_tensor = spectrogram
            else:
                # MFCC (optional)
                mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
                log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
                mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
                mfccs = mfccs[..., :num_coefficients]
                mfccs = tf.reshape(mfccs, [1, num_frames, num_coefficients, 1])
                input_tensor = mfccs

            if model is not None:
                interpreter.set_tensor(input_details[0]['index'], input_tensor)
                start_inf = time.time()
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])

            end = time.time()
            tot_latency.append(end - start)

            if model is None:
                start_inf = end

            inf_latency.append(end - start_inf)
            time.sleep(0.1)

        print('Inference Latency {:.2f}ms'.format(np.mean(inf_latency)*1000.))
        print('Total Latency {:.2f}ms'.format(np.mean(tot_latency)*1000.))
        inf = np.mean(inf_latency)*1000.
        tot = np.mean(tot_latency)*1000.
        return inf, tot
    
    
    



    


