import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#--------------------------------------------------------------

#classes:

#window generator class:
class WindowGenerator:
    def __init__(self, input_width, output_width, mean, std):
        self.input_width = input_width
        self.output_width = output_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        inputs = features[:, :-self.output_width, :]

        labels = features[:, -self.output_width, :]

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.output_width, 2])

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
                sequence_length=input_width+output_width,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess)
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds


#Multi output class for MAE:
class MultipleOutputforMAE(tf.keras.metrics.Metric):
  def __init__(self, name='MAE', **kwargs):
      super().__init__(name=name, **kwargs)
      self.total = self.add_weight(name='total', initializer='zeros',shape=(2,))
      self.count = self.add_weight('count',initializer = 'zeros')
  def update_state(self, y_true, y_pred, sample_weight=None):
    error = tf.abs(y_pred - y_true)
    error = tf.reduce_mean(error,axis = 0)
    self.total.assign_add(error)
    self.count.assign_add(1.)

  def reset_states(self):
    self.count.assign(tf.zeros_like(self.count))
    self.total.assign(tf.zeros_like(self.total))
    return

  def result(self):
    result = tf.math.divide_no_nan(self.total,self.count)
    return result

#----------------------------------------------------------------------------------


#input arguments:
parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='model name')
parser.add_argument('--number_epochs', type=int, required=False, default=1, help='number of epochs')
args = parser.parse_args()

#----------------------------------------------------------------------------------

#defining changes for different versions:
if(args.version == 'a'):
    output_width = 3
    
    #mlp model:
    MLPmodel = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2)
    ])

    #cnn model:
    CNNmodel = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=3,activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
    ])

    #LSTM model:
    LSTMmodel = keras.Sequential([
    keras.layers.LSTM(units=64),
    keras.layers.Flatten(),
    keras.layers.Dense(1)
    ])

elif(args.version == 'b'):
    output_width = 9

    #mlp model:
    MLPmodel = keras.Sequential([
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2)
    ])

    #cnn model:
    CNNmodel = keras.Sequential([
    keras.layers.Conv1D(filters=64, kernel_size=3,activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
    ])

    #LSTM model:
    LSTMmodel = keras.Sequential([
    keras.layers.LSTM(units=64),
    keras.layers.Flatten(),
    keras.layers.Dense(1)
    ])

#-----------------------------------------------------------------------------------

#main code:

#defining seed to have consistent results
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#reading the files:
csv_path = 'jena_climate_2009_2016.csv'
df = pd.read_csv(csv_path)
column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

#sperating train,val,test:
n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]

#defining important variables:
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
input_width = 6

#make the datasets:
generator = WindowGenerator(input_width, output_width, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)

#choose the metric system:
metrics = [MultipleOutputforMAE()]


#compiling models:
MLPmodel.compile(optimizer='adam',
              loss=tf.keras.losses.MAE,
              metrics=metrics)

CNNmodel.compile(optimizer='adam',
              loss=tf.keras.losses.MAE,
              metrics=metrics)

LSTMmodel.compile(optimizer='adam',
              loss=tf.keras.losses.MAE,
              metrics=metrics)



#running the models:
print("starting MLP")
MLPmodel.fit(train_ds, epochs=args.number_epochs)
print("MLP done")
print("starting CNNmodel")
CNNmodel.fit(train_ds, epochs=args.number_epochs)
print("CNNmodel done")
print("starting CNNmodel")
LSTMmodel.fit(train_ds, epochs=args.number_epochs)
print("LSTMmodel done")


#evaluating MSE:
test_loss, test_acc = MLPmodel.evaluate(test_ds, verbose=0)
print('\n MSE for MLP :', test_acc)

test_loss, test_acc = CNNmodel.evaluate(test_ds, verbose=0)
print('\n MSE for CNN :', test_acc)

test_loss, test_acc = LSTMmodel.evaluate(test_ds, verbose=0)
print('\n MSE for LSTM :', test_acc)

#MLPmodel.summary()

#save models:
run_model = tf.function(lambda x: MLPmodel(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
MLPmodel.save('MLP', signatures=concrete_func)

run_model = tf.function(lambda x: CNNmodel(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
MLPmodel.save('CNN', signatures=concrete_func)

run_model = tf.function(lambda x: LSTMmodel(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
MLPmodel.save('LSTM', signatures=concrete_func)
