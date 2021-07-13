# jetson_yolov5_tensorrt
This repo provide you easy way to convert [yolov5 model by ultralitics](https://github.com/ultralytics/yolov5) to TensorRT and fast inference wrapper. 


[TensorRT](https://developer.nvidia.com/tensorrt) - is a toolset, that contains model optimizer and high performance inference.

# Demo
VIDEO


## Requirements 
* Nvidia Jetson platform (I'm testsed on jetson nno 2gb)
* Nvidia JetPack (I'm stested on Jetpack 4.5.1)
* USB wevcamera for webcamera demo


## Performance
*The table provide number of frame per second for Jetson Nano 2GB.*
| model / image_size|    256 |        320 |     640.    |
| ----------- | -----------|----------- | ----------- |
| yolov5s + fp16  | -      | 25 fps      | 9 fps       |
| yolov5m  + fp16   | -      | -      | -       |
| yolov5l  + fp16   | -      | -      | -       |
| yolov5z  + fp16   | -      | -      |   -          |
| yolov5s     | -      | -      | 7 fps       |
| yolov5m     | -      | -      | -       |
| yolov5l     | -      | -      | -       |
| yolov5z     | -      | -      |   -          |


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


## How to build own docker container

