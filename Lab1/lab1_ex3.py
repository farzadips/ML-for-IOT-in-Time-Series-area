import argparse
import os
import picamera
import time


parser = argparse.ArgumentParser()
parser.add_argument('--width', type=int, help='width')
parser.add_argument('--height', type=int, help='height')
parser.add_argument('--fps', type=float, help='framerate')
parser.add_argument('--fmt', type=str, help='output format (png or jpg)')
parser.add_argument('-n', type=int, help='number of pictures')
parser.add_argument('-o', type=str, help='output directory')
args = parser.parse_args()


resolution = (args.width, args.height)
sleep_time = 1. / args.fps

if not os.path.exists(args.o):
    os.makedirs(args.o)

camera = picamera.PiCamera(resolution=resolution)

for i in range(args.n):
    output_file = os.path.join(args.o, '{:06}.{}'.format(i, args.fmt))
    start = time.time()
    camera.capture(output_file)
    end = time.time()
    execution_time = end - start
    size = os.path.getsize(output_file) / 2.**20 
    print('{:.1f}s, {:.2f}MB'.format(execution_time, size))
    time.sleep(sleep_time - execution_time)

camera.close()
