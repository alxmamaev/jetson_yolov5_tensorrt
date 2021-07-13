import cv2
import time
from yolov5_trt import Yolov5TRTWrapper 
from argparse import ArgumentParser


def parse():
    parser = ArgumentParser()
    parser.add_argument("engine_path")
    parser.add_argument("--labels_file")
    parser.add_argument("--source", default="0")
    parser.add_argument("--conf_th", default=0.45, type=float)

    return parser.parse_args()



def webcam_example(wrapper, device_id):
    t = time.time()
    for image, bboxes in wrapper.detect_from_webcam(device_id):
        fps = int(1 / (time.time() - t))
        image = wrapper.draw_detections(image, bboxes, fps=fps)


        cv2.imshow("demo", image)
        if cv2.waitKey(1) == 27:
            break

        t = time.time()


def images_example(wrapper, images_paths):
    images = (cv2.imread(image_path) for image_path in images_paths)
    
    for image, bboxes in wrapper.detect_from_itterator(images):
        image = wrapper.draw_detections(image, bboxes)

        cv2.imshow("demo", image)
        cv2.waitKey()


def video_examples(wrapper, video_path):
    video = cv2.VideoCapture(args.source)
    t = time.time()
    for image, bboxes in wrapper.detect_from_video(video):
        fps = int(1 / (time.time() - t))
        image = wrapper.draw_detections(image, bboxes, fps=fps)

        cv2.imshow("demo", image)
        if cv2.waitKey(1) == 27:
            break

        t = time.time()


def read_labels(labels_file):
    labels = []
    with open(labels_file) as f:
        for line in f:
            line = labels.strip()

            if not line:
                break

            labels.append(line)


def main(args):
    if args.labels_file:
        labels = read_labels(args.labels_file)
    else:
        labels = None 

    wrapper = Yolov5TRTWrapper(args.engine_path, labels=labels, conf_thresh=args.conf_th)

    if args.source.isdigit():
        device_id = int(args.source)
        webcam_example(wrapper, device_id)
    elif args.source.endswith(".mp4"):
        video_examples(wrapper, args.source)
    else:
        paths = args.source.split(",")
        
        for path in images_paths:
            assert path.endswith(".png") or path.endswith(".jpg"), f"unknown source type for file {path}"

        images_example(wrapper, images_paths)


if __name__ == "__main__":
    args = parse()
    main(args)
