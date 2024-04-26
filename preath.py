import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
import time
from pymongo import MongoClient
from datetime import datetime
from pytz import timezone
import base64
import torch
from flask import Flask, Response
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
torch.cuda.set_device(0)

# กำหนดข้อมูลเชื่อมต่อ MongoDB
mongo_uri = "mongodb+srv://myadmin:kasidate01@mycluster.puhoukq.mongodb.net/?retryWrites=true&w=majority&appName=myCluster"
database_name = "SaveImages"
collection_name = "Images"
# สร้าง MongoClient
client = MongoClient(mongo_uri)
database = client[database_name]
collection = database[collection_name]

cooldown_period = 10  # 1 minute

ZONE_POLYGON = np.array([
    [0.1, 0.1],
    [0.9, 0.1],
    [0.9, 0.9],
    [0.1, 0.9],
    [0.1, 0.1]
])

model_path = "D:\\Private\\Y3Project\\python_project\\runs\\detect\\train\\weights\\best.pt"
#rstp_url = 'rtsp://user:user@192.xxx.xx.x:554/'
myvdo = "D:\Private\Y3Project\Filter_Pic\ss100met.mp4"

app = Flask(__name__)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def save_image_to_mongodb(image_binary, count_no_helmet, count_rider):
    # Convert time to Thailand timezone
    thai_timezone = timezone('Asia/Bangkok')
    upload_time = datetime.now(thai_timezone).strftime('%Y-%m-%d')

    # Save image and time in MongoDB
    image_data = {
        "image": base64.b64encode(image_binary).decode('utf-8'),
        "upload_time": upload_time,
        "count_no_helmet": count_no_helmet,
        "count_rider": count_rider
    }

    result = collection.insert_one(image_data)
    print(f"Image uploaded successfully. Object ID: {result.inserted_id}")


def gen_frames():
    last_save_time = time.time()
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    model = YOLO(model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    box_annotator = sv.RoundBoxAnnotator(
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        text_position=sv.Position.TOP_CENTER,
        text_thickness=2,
        text_scale=1
    )

    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon, frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_thickness=4,
        text_scale=2
    )

    # Initialize FPS display text
    fps_text = "FPS: calculating..."

    fps_start_time = time.time()
    fps_counter = 0
    
    while True:
        ret, frame = cap.read()
        fps_counter += 1
        bg_color = (0, 0, 0)  # Black background
        text_color = (255, 255, 255)  # White text

        for result in model.track(source=frame, stream=True, persist=True):
            frame = result.orig_img
            selected_classes = [0, 1, 2]
            detections = sv.Detections.from_ultralytics(result)
            #detections = detections[np.isin(detections.class_id,selected_classes)]
            #mask = zone.trigger(detections=detections)

            if result.boxes.id is not None:
                detections.tracker_id = result.boxes.id.cpu().numpy().astype(int)
                
            detections = detections[np.isin(detections.class_id,selected_classes)]

            labels = []
            if detections.tracker_id is not None:
                labels = [
                    f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
                    for class_id, confidence, tracker_id
                    in zip(detections.class_id, detections.confidence, detections.tracker_id)

                ]

            if labels:
                frame = box_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections
                )
                frame = label_annotator.annotate(
                    scene=frame.copy(),
                    detections=detections,
                    labels=labels
                )
            else:
                print("No Labels")

            mask = zone.trigger(detections=detections)
            frame = zone_annotator.annotate(scene=frame)
            
            # ตรวจสอบว่า detections มีขนาดไม่เท่ากับ 0
            if len(detections) > 0:
                count_no_helmet = np.count_nonzero((detections.class_id == 0) & (detections.confidence > 0.5) & mask) #No Helmet
                count_rider = np.count_nonzero((detections.class_id == 2) & (detections.confidence > 0.5) & mask) #Rider

                # Print ค่าจำนวนออกมา
                print(f"จำนวนคนไม่สวมหมวก: {count_no_helmet}")
                print(f"จำนวนคนผู้ขับขี่: {count_rider}")

                if count_no_helmet >= 1 and count_rider >= 1:
                    current_time = time.time()
                    if current_time - last_save_time >= cooldown_period:
                        image_binary = cv2.imencode('.jpg', frame)[1].tobytes()
                        save_image_to_mongodb(image_binary, count_no_helmet, count_rider)
                        print("Save Images Successfully")
                        last_save_time = current_time
            else:
                count_no_helmet = 0
                count_rider = 0
                
            ret, buffer = cv2.imencode('.jpg',frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
        # Update FPS text
        fps_counter += 1
        if time.time() - fps_start_time >= 1:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
            fps_text = f"FPS: {fps:.2f}"     
            
        cv2.rectangle(frame, (10, 10), (200, 50), bg_color, -1)
        cv2.putText(frame, fps_text, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
        cv2.imshow("yolov8", frame)
        if (cv2.waitKey(30) == 27):
            break

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')
def index():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    #main()
    app.run(debug=True)