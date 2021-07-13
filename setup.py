from setuptools import setup

setup(
    name="yolov5_trt",
    version="0.0.2",
    author="Alexander Mamaev",
    author_email="alxmamaev@ya.ru",
    packages=["yolov5_trt"],
    scripts=["examples/yolov5_detect.py"]
)
