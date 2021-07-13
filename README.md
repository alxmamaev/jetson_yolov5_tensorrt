# jetson_yolov5_tensorrt
This repo provide you easy way to convert [yolov5 model by ultralitics](https://github.com/ultralytics/yolov5) to TensorRT and fast inference wrapper. 


[TensorRT](https://developer.nvidia.com/tensorrt) - is a toolset, that contains model optimizer and high performance inference.

# Demo
VIDEO


## Requirements 
* Nvidia Jetson platform (I'm testsed on jetson nno 2gb)
* Nvidia JetPack (I'm stested on Jetpack 4.5.1)
* USB wevcamera for webcamera demo

### Performance

| model / image_size|    256 |        320 |     640.    |
| ----------- | -----------|----------- | ----------- |
| yolov5s     | -      | 25 fps      | 7 fps       |
| yolov5m     | -      | -      | -       |
| yolov5l     | -      | -      | -       |
| yolov5z     | -      | -      |   -          |


## How to convert yolov5 model
Process of model convertation to TensorRT looks like: *Pytorch -> ONNX -> TensorRT*.
<br>Ultralitics repo already provide tool for convertation yolo to ONNX, please follow [this recipe](https://github.com/ultralytics/yolov5/issues/251).


## Run simple examples
JetPack already includes nvidia docker, you does need to install additional sofrware to run exampels.

* Pool docker container: `docker pull ...`
* Allow docker use Xserver for drawing window with dertections: `xhost +`
* Check what is your webcamera device index, by `find /dev -name video*` and find files like `/dev/video0` 
* Run webcam demo: `docker run --rm --device=/dev/video0`
- `--rm`  says remove container after exist
- `--device` says what the device you want provide inside docker container, in this case `/dev/video0` means webcamera device


## How to use wrapper in your projects


## How to build own docker container

