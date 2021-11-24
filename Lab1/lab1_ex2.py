import argparse
import os
import pyaudio
import time
import wave


parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=int, help='sampling rate in Hz')
parser.add_argument('--resolution', type=int, help='resolution (8, 16, 32)')
parser.add_argument('-o', type=str, help='output')
args = parser.parse_args()


if args.resolution == 8:
    resolution = pyaudio.paInt8
elif args.resolution == 16:
    resolution = pyaudio.paInt16
elif args.resolution == 32:
    resolution = pyaudio.paInt32
else:
    raise ValueError

samp_rate = args.rate # sampling rate
chunk = 1000 
record_secs = 3 # seconds to record
dev_index = 1 # device index found by p.get_device_info_by_index(ii)
wav_output_filename = args.o # name of .wav file

audio = pyaudio.PyAudio() # create pyaudio instantiation

# create pyaudio stream
stream = audio.open(format=resolution, rate=samp_rate, channels=1,
                    input_device_index=dev_index, input=True,
                    frames_per_buffer=chunk)
print("recording")
frames = []

# loop through stream and append audio chunks to frame array
start = time.time()
for ii in range(0, int((samp_rate / chunk) * record_secs)):
    data = stream.read(chunk)
    frames.append(data)
end = time.time()
sensing_time = end - start

print("finished recording")

# stop the stream, close it, and terminate the pyaudio instantiation
stream.stop_stream()
stream.close()
audio.terminate()

# save the audio frames as .wav file
start = time.time()
wavefile = wave.open(wav_output_filename,'wb')
wavefile.setnchannels(1)
wavefile.setsampwidth(audio.get_sample_size(resolution))
wavefile.setframerate(samp_rate)
wavefile.writeframes(b''.join(frames))
wavefile.close()
end = time.time()
storage_time = end - start

size = os.path.getsize(wav_output_filename) / 2.**10 

print('Sensing Time: {:.3f}s'.format(sensing_time))
print('Storage Time: {:.3f}s'.format(storage_time))
print('Size: {:.2f}KB'.format(size))
