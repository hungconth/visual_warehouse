from fastapi import FastAPI, WebSocket
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import asyncio
from starlette.websockets import WebSocketDisconnect
from io import BytesIO
from fastapi import HTTPException
from PIL import Image
import base64
from fastapi import Request
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import uvicorn
import json

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

# cap = cv2.VideoCapture("rtsp://admin:@192.168.0.189:554")

color = (255, 255, 255)  # BGR
thickness = 2
fontscale = 0.5


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def is_inside_quadrilateral(point, quadrilateral_vertices):
    # Check if a point is inside a quadrilateral using the winding number algorithm.
    # `point` is a tuple (x, y), and `quadrilateral_vertices` is a list of tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    x, y = point
    vertices = quadrilateral_vertices
    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    if x >= x1 and y >= y1 and x < x2 and y < y2:
        return True
    
def count_bounding_boxes_in_quadrilateral(bounding_boxes, quadrilateral_vertices):
    count = 0
    for box in bounding_boxes:
        center_x = (box[0] + box[2]) / 2  # X-coordinate of the center
        center_y = (box[1] + box[3]) / 2  # Y-coordinate of the center
        center_point = (center_x, center_y)

        if is_inside_quadrilateral(center_point, quadrilateral_vertices):
            count += 1

    return count

# Example usage:
quadrilateral_vertices1 = [(542, 338), (890, 715)]
quadrilateral_vertices2 = [(1134, 499), (1301, 743)]

count1 = 0
count2 = 0
def calculate_iou(box1, box2):
    # box1 và box2 là danh sách [left, top, right, bottom]
    left1, top1, right1, bottom1 = box1
    left2, top2, right2, bottom2 = box2
    
    # Tính diện tích của cả hai bounding boxes
    area1 = (right1 - left1) * (bottom1 - top1)
    area2 = (right2 - left2) * (bottom2 - top2)

    # Tính diện tích của phần giao nhau giữa hai bounding boxes
    x_intersection = max(0, min(right1, right2) - max(left1, left2))
    y_intersection = max(0, min(bottom1, bottom2) - max(top1, top2))
    intersection_area = x_intersection * y_intersection

    # Tính IoU
    iou = intersection_area / (area1 + area2 - intersection_area)
    return iou

model = YOLO("best.pt")
model_detect_person = YOLO("detect_person.pt")

cnt = 1

async def detect_frame():
    while True:

        cap = cv2.VideoCapture("rtsp://admin:@192.168.0.189:554")
        # await asyncio.sleep(0.05)
        ret, frame = cap.read()

        if not ret:
            break
        frame = cv2.resize(frame,(1920,1080))

        results = model(frame, imgsz=960)[0]

        # detect person
        results_person = model_detect_person(frame, save=True, imgsz=640,  classes=0)[0]


        for i in range(len(results.boxes.xyxy)):
            plot_one_box(results.boxes.xyxy[i], frame, color=(255,0,0), label=f'{results.boxes.conf[i]:.2f}')

        image1 = np.ones((1080, 1800, 3), dtype=np.uint8) * 255  # Màu trắng1

        ok1 = 0
        for i in range(len(results_person.boxes.xyxy)):
            if calculate_iou(results_person.boxes.xyxy[i],[542, 338, 890, 715]) > 0.1:
                ok1 = 1
        if ok1 == 0:
            count1 = count_bounding_boxes_in_quadrilateral(results.boxes.xyxy,quadrilateral_vertices1)
        else:
            count1 = count1
        plot_one_box([542, 338, 890, 715], frame, color=(0,0,0), label=f'')

        ok2 = 0
        for i in range(len(results_person.boxes.xyxy)):
            plot_one_box(results_person.boxes.xyxy[i], frame, color=(0,0,255), label=f'{results.boxes.conf[i]:.2f}')
            if calculate_iou(results_person.boxes.xyxy[i],[1134, 499, 1301, 743]) > 0.1:
                ok2 = 1
        if ok2 == 0:
            count2 = count_bounding_boxes_in_quadrilateral(results.boxes.xyxy,quadrilateral_vertices2)
        else:
            count2 = count2
        plot_one_box([1134, 499, 1301, 743], frame, color=(0,0,0), label=f'')

        # Thêm thông tin vào ảnh
        cv2.putText(image1, f"linh_kien1 : {count1}", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)
        cv2.putText(image1, f"linh_kien2 : {count2}", (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), thickness=5, lineType=cv2.LINE_AA)

        # Tạo hình vuông màu trắng 300x300
        image2 = frame

        # Ghép hai hình ảnh theo chiều ngang để tạo image3
        image3 = np.hstack((image2,image1))


        frame = image3
        frame = cv2.resize(frame,(1280,720))

        # convert RGB to BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Convert frame to base64 for displaying in the browser
        frame_pil = Image.fromarray(frame)
        buffered = BytesIO()
        frame_pil.save(buffered, format="JPEG")
        frame_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        # response = json.dumps({"bounding_boxes": bounding_boxes, "labels": labels})
        yield frame_base64


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    try:
        async for frame_base64 in detect_frame():
            await websocket.send_text(frame_base64)
    except WebSocketDisconnect:
        pass

@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run("app:app", host="localhost", port=8000, reload=True)