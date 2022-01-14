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
import tensorflow_model_optimization as tfmot

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version')
args = parser.parse_args()





class read_audios:
    def __init__(self):
        pass
    
    def read(self):
        train_file = open('kws_train_split.txt', "r") #opens the file in read mode
        train_lines = train_file.read().splitlines() #puts the file into an array
        train_file.close()
        for i in range(len(train_lines)):
            train_lines[i] = train_lines[i][2:]
        train_tf = tf.convert_to_tensor(train_lines)
        
        test_file = open('kws_test_split.txt', "r") #opens the file in read mode
        test_lines = test_file.read().splitlines() #puts the file into an array
        test_file.close()
        for i in range(len(test_lines)):
            test_lines[i] = test_lines[i][2:]
        test_tf = tf.convert_to_tensor(test_lines)
        
        val_file = open('kws_val_split.txt', "r") #opens the file in read mode
        val_lines = val_file.read().splitlines() #puts the file into an array
        val_file.close()
        for i in range(len(val_lines)):
            val_lines[i] = val_lines[i][2:]
        val_tf = tf.convert_to_tensor(val_lines)
        
        return train_tf, val_tf, test_tf





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

    def models(self,alpha, strides, units, mfcc, mymodel, magnitude = False, train_ds = None):
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


        MODELS = {'mlp'+ mymodel : mlp, 'cnn'+ mymodel: cnn, 'ds_cnn'+ mymodel: ds_cnn}
        model = MODELS['ds_cnn' + mymodel] 
        loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
        optimizer = tf.optimizers.Adam()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]
        model.compile(loss = loss, optimizer = optimizer, metrics = metrics)

        if mfcc is False:
            checkpoint_filepath = f'./checkpoints/stft/chkp_best_{mymodel}'

        else:
            checkpoint_filepath = f'./checkpoints/mfcc/chkp_best_{mymodel}'
        if not magnitude:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_filepath,           
                monitor='val_sparse_categorical_accuracy',
                verbose=1,
                mode='max',
                save_best_only=True,
                save_freq='epoch')
        else:
            pruning_params = {'pruning_schedule':
            tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.30,
            final_sparsity=0.8,
            begin_step=len(train_ds)*5,
            end_step=len(train_ds)*15)
            }
            prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
            model = prune_low_magnitude(model, **pruning_params)
            model_checkpoint_callback = tfmot.sparsity.keras.UpdatePruningStep()
        return model, model_checkpoint_callback, checkpoint_filepath




    

class model_analysis():
    def __init__(self, test_ds, checkpoint_filepath, train_ds):
        self.test_ds = test_ds
        self.checkpoint_filepath = checkpoint_filepath
        self.train_ds = train_ds

    def S_pruning_Model_evaluate_and_compress_to_TFlite(self, tflite_model_dir):
        if not os.path.exists('./models'):
            os.makedirs('./models')
        if not os.path.exists('./compressed_models/'):
            os.makedirs('./compressed_models/')
        best_model = tf.keras.models.load_model(filepath = self.checkpoint_filepath )
        Loss , ACCURACY = best_model.evaluate(self.test_ds)
        # Convert to TF lite without Quantization 
        converter = tf.lite.TFLiteConverter.from_saved_model(self.checkpoint_filepath)
        tflite_model = converter.convert()  
        Compressed = "compressed_"+tflite_model_dir 
        tflite_model_dir = './models/'+tflite_model_dir
        # Write the model in binary formate and save it 
        with open(tflite_model_dir, 'wb') as fp:
            fp.write(tflite_model)
        Compressed = './compressed_models/'+Compressed
        with open(Compressed, 'wb') as fp:
            tflite_compressed = zlib.compress(tflite_model)
            fp.write(tflite_compressed)
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

        inf = np.mean(inf_latency)*1000.
        tot = np.mean(tot_latency)*1000.
        return inf, tot

####################
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
import time
print(f"Python version used to excute the code is {python_version()}")
import warnings
warnings.filterwarnings('ignore')
epochs = 20
mymodel = "model_"
TFLITE =  f'{mymodel}.tflite'     # path for saving the best model after converted to TF.lite model 
units = 8                         # The number of output class [8:without silence , 9 : with silence]
################## Fix the Random seed to reproduce the same results 
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
#zip_path = tf.keras.utils.get_file(
#     origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
#     fname='mini_speech_commands.zip',
#     extract=True,
#     cache_dir='.', cache_subdir='data')
#data_dir = os.path.join('.', 'data', 'mini_speech_commands')
reading_class = read_audios()
train_files, val_files, test_files = reading_class.read()
"""alpha = [0.59]
mfcc = [True]
m = ['ds_cnn']
number_of_bins = [16]
lower_freq = [20]
upper_freq = [4000]
sample_rate = [16000]
frame_length = [1000]
frame_step = [500]"""

if(str(args.version) == 'a'):
    alpha = [0.59]
    mfcc = [True]
    m = ['ds_cnn']
    number_of_bins = [16]
    lower_freq = [20]
    upper_freq = [4000]
    sample_rate = [16000]
    frame_step = [350]
    frame_length = [1000]
elif(str(args.version) == 'b'):
    alpha = [0.6]
    mfcc = [True]
    m = ['ds_cnn']
    number_of_bins = [16]
    lower_freq = [20]
    upper_freq = [4000]
    sample_rate = [16000]
    frame_step = [256]
    frame_length = [1024]
elif(str(args.version) == 'c'):
    alpha = [0.3]
    mfcc = [True]
    m = ['ds_cnn']
    number_of_bins = [16]
    lower_freq = [20]
    upper_freq = [4000]
    sample_rate = [16000]
    frame_step = [256]
    frame_length = [1024]


df = pd.DataFrame(columns=['alpha', 'mfcc', 'model', 'number_of_bins', 'lower_freq', 'upper_freq','sample_rate','frame_step', 'frame_length','n_acuracy','n_size','n_latency','q_acuracy','q_size','q_latency'])
for p_alpha in range(len(alpha)):
    for p_mfcc in range(len(mfcc)):
        for p_m in range(len(m)):
            for p_bins in range(len(number_of_bins)):
                for p_lower_f in range(len(lower_freq)):
                    for p_upper_f in range(len(upper_freq)):
                        for p_rate in range(len(sample_rate)):
                            for p_f_step in range(len(frame_step)):
                                for p_f_length in range(len(frame_length)):
                                    if((lower_freq[p_lower_f] < upper_freq[p_upper_f]) and (upper_freq[p_lower_f] <= sample_rate[p_upper_f]/2) and (frame_step[p_f_step] < frame_length[p_f_length])):
                                        data = [[alpha[p_alpha],mfcc[p_mfcc],m[p_m],number_of_bins[p_bins],lower_freq[p_lower_f],upper_freq[p_upper_f],sample_rate[p_rate],frame_step[p_f_step],frame_length[p_f_length],0,0,0,0,0,0]]
                                        d = pd.DataFrame(data, columns=['alpha', 'mfcc', 'model', 'number_of_bins', 'lower_freq', 'upper_freq','sample_rate','frame_step', 'frame_length','n_acuracy','n_size','n_latency','q_acuracy','q_size','q_latency'])
                                        df = df.append(d, ignore_index = True)

LABELS = np.array(['stop', 'up', 'yes', 'right', 'left', 'no',  'down', 'go'] , dtype = str) 
start = 0
end = 0
for i in range(df.shape[0]):
    print("configuration  " + str(i) + " from  " + str(df.shape[0])+ "!!!!!!!!!!!!")
    print("time estimate is:    " + str((end - start)/60 *(df.shape[0] - i)) + " minutes left")
    start = time.time()
    
    mymodel = "V_" + str(i)
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
    model, model_checkpoint_callback, checkpoint_filepath = model_maker.models(df['alpha'][i], strides, units, df['mfcc'][i], mymodel,False,train_ds)
    ############ Applying Magnitude-Based Pruning
    #model, model_checkpoint_callback, checkpoint_filepath = model_maker.models(1, strides, units, model_version, mfcc, mymodel)
    history = model.fit(train_ds, epochs=epochs,   validation_data=val_ds,callbacks=[model_checkpoint_callback ], verbose=0)
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
    end = time.time()
    


