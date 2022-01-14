import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow.lite as tflite
from tensorflow import keras
import zlib
import argparse
import tensorflow_model_optimization as tfmot   

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version')
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

print (mean.shape)
version = args.version
input_width = 6
if args.version == "a" :                
    output_steps = 3
if args.version == "b" :
    output_steps = 9


class WindowGenerator:
    def __init__(self, input_width, output_steps, mean, std):
        self.input_width = input_width
        self.output_steps = output_steps
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])


    def split_window(self, features):
        inputs = features[:, :self.input_width, :]        # for example if total window size = 9 input =  [:,:6 ,:] --> output [:,-3: , ] outpu_tstep = 3 
        labels = features[:, -self.output_steps :, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_steps,2])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length = input_width + self.output_steps,                       #### this change because now the total depends on the output widht
                sequence_stride = 1,
                batch_size = 32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

generator = WindowGenerator(input_width, output_steps, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

print(f"data shape before split {data.shape}")

print(f"train_data {train_data.shape}")

print(f"val_data {val_data.shape}")

print(f"test_data {test_data.shape}")


# # Exercise 1.7
class MultiOutputMAE(tf.keras.metrics.Metric):
    def __init__(self, name='mean_absolute_error', **kwargs):
        super().__init__(name=name, **kwargs)
        self.total = self.add_weight('total', initializer='zeros', shape=(2,))
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.abs(y_pred- y_true)     
        error = tf.reduce_mean(error, axis=[0,1])  # compute the mean for all samples in the batch for each feature (temp , hum) ==> output shape = (2,)
        self.total.assign_add(error)
        self.count.assign_add(1.)

        return

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))

    def result(self):
        result = tf.math.divide_no_nan(self.total, self.count)

        return result

alpha = 0.03
# alpha = 1
sparsity = 0.9
Structured = True
if Structured == True :
    model_version = f"_V( {version}_alpha={alpha} )"
else :
    model_version = f"_V( {version}_Sparcity ={sparsity} )"
mymodel = 'mlp'+ model_version
chk_path = f'./callback_{mymodel}_chkp/{mymodel}_chkp_best'     # path for saving the best model 
TFLITE = mymodel + ".tflite"                                    # path for saving the best model after converted to TF.lite model 

print(mymodel)



#################################### Build models with Structured Pruning via width Multiplier #####################################################
def bulid_models_Structured (alpha = alpha , version = version , input_width = input_width ,output_steps = output_steps ,model_version = model_version  ) :
    mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = (input_width,2) , name='Flatten'),
            tf.keras.layers.Dense(int(128 *alpha), activation='relu' , name='Dense1'),
            tf.keras.layers.Dense(int(128 *alpha), activation='relu' , name='Dense2'),
            tf.keras.layers.Dense(units = 2*output_steps , name='Output_layes'), 
            tf.keras.layers.Reshape([output_steps, 2])
        ])

    ############################################################################
    cnn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(input_shape = (input_width,2) , filters=int(64 *alpha), kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=int(64 *alpha), activation='relu'),
            tf.keras.layers.Dense(units=2*output_steps), 
            tf.keras.layers.Reshape([output_steps, 2])
        ])

  

    MODELS = {'mlp'+ model_version: mlp, 'cnn'+ model_version: cnn }
    return MODELS 

####################################       Build models with UnStructured Pruning         #####################################################

def bulid_models_UnStructured ( version = version , input_width = input_width ,output_steps = output_steps ,model_version = model_version  ) :
    mlp = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape = (input_width,2) , name='Flatten'),
            tf.keras.layers.Dense(128, activation='relu' , name='Dense1'),
            tf.keras.layers.Dense(128, activation='relu' , name='Dense2'),
            tf.keras.layers.Dense(units = 2*output_steps , name='Output_layes'), 
            tf.keras.layers.Reshape([output_steps, 2])
        ])

    ############################################################################
    cnn = tf.keras.Sequential([
            tf.keras.layers.Conv1D(input_shape = (input_width,2) , filters=64, kernel_size=3, activation='relu' , name= "Conv1D-1"),
            tf.keras.layers.Flatten(name='Flatten'),
            tf.keras.layers.Dense(units=64, activation='relu', name='Dense-1'),
            tf.keras.layers.Dense(units=2*output_steps), 
            tf.keras.layers.Reshape([output_steps, 2])
        ])

 

    MODELS = {'mlp'+ model_version: mlp, 'cnn'+ model_version: cnn }
    return MODELS 

if Structured == True :
    MODELS = bulid_models_Structured()
else :
    MODELS = bulid_models_UnStructured()
    
def get_model(model = mymodel ):
    model = MODELS[model]
    loss =   tf.keras.losses.MeanSquaredError()                       #tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    metrics = [MultiOutputMAE()]

    # Training and optimizing

    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    
    return model


# Create TEMP_HUM_VAL  callback to print the MAE for Temperature and humidity in more interpetable format 
class TEMP_HUM_VAL(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        hum = logs["val_mean_absolute_error"][1]
        temp = logs["val_mean_absolute_error"][0]
        MAE = logs["val_mean_absolute_error"]
        print(f"\n Temp MAE = {temp:.3f}, Hum MAE = {hum:.3f}    ")
        # return temp , hum


mycallback = TEMP_HUM_VAL()


# Create checkpoint callback to save the best model 
cp_callback = keras.callbacks.ModelCheckpoint(
    f'./callback_{mymodel}_chkp/{mymodel}_chkp_best',
    # './callback_test_chkp/chkp_best',
    monitor='val_loss',
    verbose=0, 
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    save_freq='epoch')

if Structured == True :
    model = get_model(mymodel)
    history = model.fit(train_ds, epochs=20,   validation_data=val_ds,callbacks=[mycallback ,cp_callback ])
if Structured == False :
# Create  Unstructiured Pruning Callback  to to apply pruning during Training and fitting the model
    model = MODELS[mymodel]
    # Define the sparsity scheduler
    pruning_params = {'pruning_schedule':
    tfmot.sparsity.keras.PolynomialDecay(
    initial_sparsity=0.30,
    final_sparsity=sparsity,
    begin_step=len(train_ds)*5,
    end_step=len(train_ds)*15)
    }

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    model = prune_low_magnitude(model, **pruning_params)
    # Define the pruning callback
    PruningCallback = [tfmot.sparsity.keras.UpdatePruningStep()]
    
if Structured == False :
    loss =   tf.keras.losses.MeanSquaredError()                       
    optimizer = tf.keras.optimizers.Adam()
    metrics = [MultiOutputMAE()]
    input_shape = [32, 6, 2]
    model.build(input_shape)
    model.compile(loss = loss, optimizer = optimizer, metrics = metrics)
    history = model.fit(train_ds, epochs=50,   validation_data=val_ds,callbacks=[ PruningCallback])

    model_to_export = tfmot.sparsity.keras.strip_pruning(model)

    
############################## Print Model Summary ####################
print(model.summary())


def Pruned_Model_evaluate_and_compress_to_TFlite(model = model , model_to_export=None,tflite_model_dir =  TFLITE , model_name = mymodel  ):
    
    if not os.path.exists('./Pruned_models'):
        os.makedirs('./Pruned_models')
    # Model Evaluation
    loss, error = model.evaluate(test_ds)
    print(f'Temp mae = {error[0]:.3f}: , HUM mae = {error[1]:.3f} ')  
    # Convert to TF lite without Quantization 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()  
    # Write the model in binary formate and save it 
    with open(tflite_model_dir, 'wb') as fp:
        fp.write(tflite_model)
    Compressed =  f"compressed_{TFLITE}"
    with open(Compressed, 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"the model is saved successfuly to {tflite_model_dir}")
    return Compressed , tflite_model_dir 

def apply_Quantization(tflite_model_dir =  TFLITE ,  PQT = False , WAPQT = False , Structured = Structured , saving_path = None ): 
    if Structured == False :
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if Structured == True :
        converter = tf.lite.TFLiteConverter.from_saved_model(saving_path)
    # Apply weight only quantization 
    if PQT == True :
        tflite_model_dir = f"PQT_{tflite_model_dir}"
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        tflite_model = converter.convert()
    # Apply weight + Activation  quantization 
    if WAPQT == True :
        converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
        converter.representative_dataset = representative_dataset_gen
        tflite_model = converter.convert()
        
        tflite_model_dir = f"WAPQT_{tflite_model_dir}"
      
    # Write the model in binary formate and save it 
    with open(tflite_model_dir, 'wb') as fp:
        fp.write(tflite_model)
    Compressed =  f"compressed_{tflite_model_dir}"
    with open(Compressed, 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"the model is saved successfuly to {tflite_model_dir}")
    return Compressed , tflite_model_dir 

# Function for weight and activations quantization 
def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]

def S_pruning_Model_evaluate_and_compress_to_TFlite( tflite_model_dir =  TFLITE , chk_path = chk_path , model_name = mymodel , PQT = False , WAPQT = False , Structured = Structured):
    if not os.path.exists('./models'):
        os.makedirs('./models')
    model = tf.keras.models.load_model(filepath = chk_path , custom_objects={'MultiOutputMAE':MultiOutputMAE})

    run_model = tf.function(lambda x: model(x))
    # input_shape = model.inputs[0].shape.as_list()
    # input_shape[0] = batch_size
    # func = tf.function(model).get_concrete_function(
    # tf.TensorSpec(input_shape, model.inputs[0].dtype))
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2], tf.float32))
    saving_path = os.path.join('.','models', model_name)
    model.save(saving_path ,signatures=concrete_func )

    best_model = tf.keras.models.load_model(filepath = saving_path , custom_objects={'MultiOutputMAE':MultiOutputMAE})
    loss, error = best_model.evaluate(test_ds)
    print(f'Temp mae = {error[0]:.3f}: , HUM mae = {error[1]:.3f} ')
    # Convert to TF lite without Quantization 
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()  

    # Write the model in binary formate and save it 
    with open(tflite_model_dir, 'wb') as fp:
        fp.write(tflite_model)
    Compressed = "compressed_"+tflite_model_dir 
    with open(Compressed, 'wb') as fp:
        tflite_compressed = zlib.compress(tflite_model)
        fp.write(tflite_compressed)
    print(f"the model is saved successfuly to {tflite_model_dir}")
    return Compressed , tflite_model_dir , saving_path

def load_and_evaluation(path, dataset):
    interpreter = tf.lite.Interpreter(model_path = path) 
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dataset = test_ds.unbatch().batch(1)

    outputs = []
    labels = []
    print(dataset)

    for data in dataset:
        my_input = np.array(data[0], dtype = np.float32)
        label = np.array(data[1], dtype = np.float32)
        # print (f"my_input = {my_input}")
        # print(f"label = {label}")

    
            
        labels.append(label)

        interpreter.set_tensor(input_details[0]['index'], my_input)
        interpreter.invoke()
        my_output = interpreter.get_tensor(output_details[0]['index'])
        
        outputs.append(my_output[0])

    outputs = np.squeeze( np.array(outputs))
    labels = np.squeeze(np.array(labels))

    
    error = np.absolute(outputs - labels)
  
    mean_axis_1 = np.mean(error , axis = 1)     #  ==>  np.sum(error, axis = 1)/labels.shape[1]
    
    mae = np.mean(mean_axis_1 , axis = 0)  #  ==> np.sum(mean_axis_1, axis = 0) /mean_axis_1.shape[0]
    temp_MAE = mae[0] 
    hum_MAE = mae[1]
    print("*"*50,"\n",f" Excuting the model {path} ")
    print("*"*50,"\n",f" mae is {mae} of shape {mae.shape} ")
    print("*"*50,"\n",f'Temp mae = {mae[0]:.3f}: , HUM mae = {mae[1]:.3f} ')
    
    # check version "a" requirment
    if version == "a" :
        if temp_MAE <= 0.3 and hum_MAE <= 1.2 :
            print ("*"*50,"\n","achieved the requirments " )
        else :
            print ("*"*50,"\n","Not achieved the requirments" )
    # check version "b" requirment        
    if version == "b" :
        if temp_MAE <= 0.7 and hum_MAE <= 2.5 :
            print ("*"*50,"\n","achieved the requirments " )
        else :
            print ("*"*50,"\n","Not achieved the requirments" ) 


def getsize(file):
    st = os.stat(file)
    size = st.st_size
    return size

if Structured == False :
    tflite_model_dir_Compressed ,tf_lite_model_path = Pruned_Model_evaluate_and_compress_to_TFlite(model_to_export=model_to_export)
if Structured == True :
    tflite_model_dir_Compressed ,tf_lite_model_path, saving_path  = S_pruning_Model_evaluate_and_compress_to_TFlite(TFLITE)
# print(b)
tflite_size = getsize(tf_lite_model_path)
compressed_tflite_size = getsize(tflite_model_dir_Compressed )
print(f" \n Size of TF.lite model is {tflite_size /1000} KB")
print(f" \n Size of compressed_tflite model is {compressed_tflite_size /1000} KB")
print(tf_lite_model_path)
load_and_evaluation(tf_lite_model_path , test_ds) 
if Structured == False :
    tflite_model_dir_Compressed ,tf_lite_model_path = apply_Quantization( PQT=True)
if Structured == True :
    Compressed , Quantized   = apply_Quantization(PQT=True ,  saving_path=saving_path)
# print(b)
tflite_size = getsize(Quantized)
compressed_tflite_size = getsize(Compressed )
print(f" \n Size of TF.lite model is {tflite_size /1000} KB")
print(f" \n Size of compressed_tflite model is {compressed_tflite_size /1000} KB")
load_and_evaluation(Quantized , test_ds) 

if Structured == False :
    tflite_model_dir_Compressed ,tf_lite_model_path = apply_Quantization( WAPQT=True)
if Structured == True :
    Compressed , Quantized   = apply_Quantization(WAPQT=True , saving_path=chk_path)
# print(b)
tflite_size = getsize(Quantized)
compressed_tflite_size = getsize(Compressed )
print(f" \n Size of TF.lite model is {tflite_size /1000} KB")
print(f" \n Size of compressed_tflite model is {compressed_tflite_size /1000} KB")
load_and_evaluation(Quantized , test_ds) 