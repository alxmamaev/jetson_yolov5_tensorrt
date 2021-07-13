import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
import cv2
from .nms import non_max_supression


def SeessionInitError(Exception):
    """Raises when trying to get a session properties before initialisation"""
    def __init__(self, property_name):
        message = f"Trying to get {property_name} before session intialisation."
        super().__init__(message)


class Yolov5TRTSession:
    def __init__(self, serialized_model):
        self.__bindings = None
        self.__inputs = None
        self.__input_shapes = None
        self.__outputs = None
        self.__output_shapes = None
        self.logger = None
        self.runtime = None
        self.engine = None
        self.stream = None
        self.context = None

        self.__serialized_model = serialized_model


    def __enter__(self):
        self.__bindings = [] 
        self.__inputs = [] 
        self.__input_shapes = [] 
        self.__outputs = []
        self.__output_shapes = [] 
        
        self.logger = trt.Logger()
        self.runtime = trt.Runtime(self.logger)
        self.stream = cuda.Stream()
        
        self.engine = self.runtime.deserialize_cuda_engine(self.__serialized_model)
        self.context = self.engine.create_execution_context()

        for binding in self.engine:
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            binding_shape = [int(i) for i in self.engine.get_binding_shape(binding)]

            mem = cuda.managed_empty(binding_shape, dtype, mem_flags=cuda.mem_attach_flags.GLOBAL)
            self.__bindings.append(int(mem.ctypes.data))
            

            if self.engine.binding_is_input(binding):
                self.__inputs.append(mem)
                self.__input_shapes.append(binding_shape)
            else:
                self.__outputs.append(mem)
                self.__output_shapes.append(binding_shape)

        return self


    @property
    def inputs_memory(self):
        if self.__inputs is None:
            raise SessionInitError("inputs")
        else:
            return self.__inputs


    @property
    def input_shapes(self):
        if self.__input_shapes is None:
            raise SessionInitError("input shapes")
        else:
            return self.__input_shapes


    def execute(self):
        self.context.execute_async_v2(bindings=self.__bindings, stream_handle=self.stream.handle)
        self.stream.synchronize()

        out = [output.reshape(shape) for output, shape in zip(self.__outputs, self.__output_shapes)]
        return out


    def __exit__(self, exc_type, exc_value, exc_tb):
        del self.context
        del self.engine
        del self.runtime
        del self.logger



class Yolov5TRTWrapper:
    def __init__(self, engine_path, conf_thresh=0.25, labels=None):
        self.conf_thresh = conf_thresh
        self.labels = labels
        with open(engine_path, "rb") as f:
            self.serialized_model = f.read()


    def create_session(self):
        return Yolov5TRTSession(self.serialized_model) 


    def resize_images(self, images, image_size, dtype=np.float32):
        resized_images = []
        scale_factors = []      

        for image in images:
            scale_factor = max(image.shape[0], image.shape[1]) / image_size
            new_size = int(image.shape[1] / scale_factor), int(image.shape[0] / scale_factor)

            resized_image = cv2.resize(image, new_size)
            resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            resized_image =  np.array(resized_image[:image_size,:image_size], dtype=dtype, order="C") / 255.0
            resized_image = np.moveaxis(resized_image, 2, 0)

            resized_images.append(resized_image)
            scale_factors.append(scale_factor)

        return resized_images, scale_factors


    def images2memory(self, images, memory):
        for i, image in enumerate(images):
            memory[i, :, :image.shape[1], :image.shape[2]] = image


    def detect_from_itterator(self, itterator):
        with self.create_session() as session: 
            for image in itterator:
                bboxes = self.detect_from_batch_images([image], session)
                yield image, bboxes[0]


    def detect_from_video(self, video):
        with self.create_session() as session:
            while True:
                ret, image = video.read()
                if not ret:
                    return

                bboxes = self.detect_from_batch_images([image], session)
                yield image, bboxes[0]
            

    def detect_from_webcam(self, device_id):
        cap = cv2.VideoCapture(device_id)
        for image, bboxes in self.detect_from_video(cap):
            yield image, bboxes


    def detect_from_batch_images(self, batch_images, session):
        model_batch_size = session.input_shapes[0][0]
        batch_size = len(batch_images)

        assert batch_size <= model_batch_size, f"Batch size must be less or equal to model batch_size, expect <={model_batch_size}, got {batch_size}"

        image_size = min(session.input_shapes[0][2:])
        memory = session.inputs_memory[0]
        batch_images, scale_factors = self.resize_images(batch_images, image_size, dtype=memory.dtype)
        
        self.images2memory(batch_images, memory)
        out = session.execute()[-1][:batch_size]
        bboxes = non_max_supression(out, conf_thresh=self.conf_thresh)

        for i, s in enumerate(scale_factors):
            bboxes[i][:,:4] *= s

        return bboxes
            

    def draw_detections(self, image, bboxes, fps=None):
        image = image.copy()
        coord = bboxes[:,:4].astype(np.int32)
        for i in range(len(coord)):
            x1, y1, x2, y2 = coord[i]
            prob = round(float(bboxes[i][4]), 2)
            class_id = int(bboxes[i][5])
            label = str(class_id) if self.labels is None else self.labels[class_id]
            
            text = f"{label} prob: {prob}"

            cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 4)
            cv2.putText(image, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        if fps is not None:
            cv2.putText(image, f"FPS: {fps}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 200, 20), 5)

        return image
