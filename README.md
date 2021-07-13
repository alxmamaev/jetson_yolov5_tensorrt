# jetson_yolov5_tensorrt
This repo provide you easy way to convert [yolov5 model by ultralitics](https://github.com/ultralytics/yolov5) to TensorRT and fast inference wrapper. 


[TensorRT](https://developer.nvidia.com/tensorrt) - is a toolset, that contains model optimizer and high performance inference.

# Demo
![VIDEO](https://www.youtube.com/watch?v=Gg_El_NgPs8)


## Requirements 
* Nvidia Jetson platform (I'm testsed on jetson nno 2gb)
* Nvidia JetPack (I'm stested on Jetpack 4.5.1)
* USB wevcamera for webcamera demo


## Performance
*The table provide number of frame per second for Jetson Nano 2GB.*
| model / image_size|    256 |        320 |     640    |
| ----------- | -----------|----------- | ----------- |
| yolov5s + fp16  | -      | 25 fps      | 9 fps       |
| yolov5m  + fp16   | -      | -      | -       |
| yolov5l  + fp16   | -      | -      | -       |
| yolov5x  + fp16   | -      | -      |   -          |
| yolov5s     | -      | -      | 7 fps       |
| yolov5m     | -      | -      | -       |
| yolov5l     | -      | -      | -       |
| yolov5x     | -      | -      |   -          |


## How to convert yolov5 model
Process of model convertation to TensorRT looks like: *Pytorch -> ONNX -> TensorRT*.
<br>Ultralitics repo already provide tool for convertation yolo to ONNX, please follow [this recipe](https://github.com/ultralytics/yolov5/issues/251).

After that you need to use `trtexec` tool, my docker container includes builded trtexec. You can use it just by pulling the container.
JetPack already includes nvidia docker, you does need to install additional sofrware to run exampels.
* Pool docker container: `docker pull ...` (if you not pull it yet)
* Run `docker run --runtime nvidia -v /path/to/dir/with/model/:/models/ --rm yolov5_trt:latest trtexec --onnx=/models/model_name.onnx --saveEngine=model_name.plan -  -fp16`
  - Provide directory with your model after `-v` option, this dir will be shared between container and the host.
  - Also replace `model_name` by name of your model file
  - TensorRT model will be saved at path that sets in `--saveEngine` option
  - If you want to know more convertion options call trtexec with `--help` option

## Run simple examples
* Pool docker container: `docker pull ...` (if you not pull it yet)
* Allow docker use Xserver for drawing window with dertections: `xhost +`
* Check what is your webcamera device index, by `find /dev -name video*` and find files like `/dev/video0` 
* Run webcam demo: `docker run --rm --device=/dev/video0`
   - `--rm`  says remove container after exist
   - `--device` says what the device you want provide inside docker container, in this case `/dev/video0` means webcamera device


## How to use wrapper in your projects

### Getting detection from the camera
```python

import cv2
from yolov5_trt import Yolov5TRTWrapper 

labels = [...] # List of class names for your model
conf_th = 0.25 # Confidence threshold

wrapper = Yolov5TRTWrapper(args.engine_path, labels=labels, conf_thresh=conf_th)
for image, bboxes in wrapper.detect_from_webcam(0): # Gets detection and image from the usb camera with id 0
    image = wrapper.draw_detections(image, bboxes) # Drawing bboxes on the image

    # Show detections in the window with name "demo"
    cv2.imshow("demo", image) 
    if cv2.waitKey(1) == 27:
        break
```

### Getting detection from the Video
```python

import cv2
from yolov5_trt import Yolov5TRTWrapper 

labels = [...] # List of class names for your model
conf_th = 0.25 # Confidence threshold

wrapper = Yolov5TRTWrapper(args.engine_path, labels=labels, conf_thresh=conf_th)

video = cv2.VideoCapture("video.mp4") # Opening video file

for image, bboxes in wrapper.detect_from_video(video): # Gets detection and image from the video
    image = wrapper.draw_detections(image, bboxes) # Drawing bboxes on the image

    # Show detections in the window with name "demo"
    cv2.imshow("demo", image) 
    if cv2.waitKey(1) == 27:
        break
```

### Getting detection from the Itterator (also generator, or list)
```python

import cv2
from yolov5_trt import Yolov5TRTWrapper 

labels = [...] # List of class names for your model
conf_th = 0.25 # Confidence threshold

wrapper = Yolov5TRTWrapper(args.engine_path, labels=labels, conf_thresh=conf_th)

images_paths = [...] # Path to images
images = (cv2.imread(image_path) for image_path in images_paths)

for image, bboxes in wrapper.detect_from_itterator(video): # Gets detection and image from the itterator
    image = wrapper.draw_detections(image, bboxes) # Drawing bboxes on the image

    # Show detections in the window with name "demo"
    cv2.imshow("demo", image) 
    cv2.waitKey()
```

### Getting detection from batch of images 
*Note: in the streaming tasks you does not need to process batches with size more than 1. Because in streaming latency is more important than throughput.*
<br>
*Note2: Be careful, the size of the input batch must be less than or equal to the maximum batch size specified during conversion*

```python

import cv2
from yolov5_trt import Yolov5TRTWrapper 

labels = [...] # List of class names for your model
conf_th = 0.25 # Confidence threshold

wrapper = Yolov5TRTWrapper(args.engine_path, labels=labels, conf_thresh=conf_th)

images_paths = [...] # Path to images
images_batch = [cv2.imread(image_path) for image_path in images_paths] # read images batch

session = wrapper.create_session()
with session:
    bboxes = wrapper.detect_from_batch_images(images_batch, session): # Gets detection from images batch.
    for b, img in zip(bboxes, images_batch)
      image = wrapper.draw_detections(img, b) # Drawing bboxes on the image

      # Show detections in the window with name "demo"
      cv2.imshow("demo", image) 
      cv2.waitKey()
```

Session creating takes a some time, for high performance, process all batch inside `with session` block (befor session closing). 

## How to build own docker container
Building possible only in nvidia runtime, to setting up nvidia runtime as default, edit `/etc/docker/daemon.json`  file.
```json
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia"
}
```
And restart docker by `sudo systemctl restart docker`. After that build docker container `docker build . -t yolo5_trt:latest`
