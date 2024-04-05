from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

openvino_model_path = 'models/best_openvino_model'
openvino_model = YOLO(openvino_model_path, task='detect')

class apdDetection:
    def __init__(self):
        pass

    def resize_image(self, image, desired_width=640):
        height, width = image.shape[:2]
        ratio = desired_width / width
        desired_height = int(height * ratio)
        image = cv2.resize(image, (desired_width, desired_height), interpolation=cv2.INTER_AREA)
        return image
    
    def detect_object(self, image):
        results = openvino_model.predict(image, conf=0.25, iou=0.45)[0]
        if results is None or len(results) == 0:
            return None
        detection_result = []
        for det in results.boxes.xyxy:
            xmin, ymin, xmax, ymax = det[:4]  # Extracting the bounding box coordinates
            detection_result.append((xmin, ymin, xmax, ymax))
        return detection_result

    def draw_bounding_boxes(self, frame, detection_result, resized_frame):
        if detection_result is not None:
            for box in detection_result:
                xmin, ymin, xmax, ymax = box  # Unpack the box coordinates
                xmin_orig = int(xmin * frame.shape[1] / resized_frame.shape[1])
                xmax_orig = int(xmax * frame.shape[1] / resized_frame.shape[1])
                ymin_orig = int(ymin * frame.shape[0] / resized_frame.shape[0])
                ymax_orig = int(ymax * frame.shape[0] / resized_frame.shape[0])
                cv2.rectangle(frame, (xmin_orig, ymin_orig), (xmax_orig, ymax_orig), (0, 255, 0), 2)
        return frame

class VideoCamera(object):
    def __init__(self):
        self.apd_detector = apdDetection()
        self.video = cv2.VideoCapture("source_files/hardhat.mp4")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, frame = self.video.read()
        if success:
            resized_frame = self.apd_detector.resize_image(frame)
            detection_result = self.apd_detector.detect_object(resized_frame)
            frame_with_boxes = self.apd_detector.draw_bounding_boxes(frame, detection_result, resized_frame)
            ret, jpeg = cv2.imencode('.jpg', frame_with_boxes)
            return jpeg.tobytes()
        else:
            return None

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
