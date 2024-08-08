from ultralytics import YOLO
import cv2

# First videos
# 7
# 18
# 36
# 24


# Second videos
# 15
# 20
# 29
# 22


# Third video
# 2
# 4
# 14
# 4

# Fourth video
# 8
# 8
# 23
# 24

# Fifth video
# 8
# 7
# 63
# 13

# Sixth video
# 9
# 15
# 13 --- : Not usable
# 14


# load yolov8 model
model = YOLO("yolov8n.pt")

# load video
video_path = "/home/ge42top/Desktop/output/1_video_5_aldm.avi"
cap = cv2.VideoCapture(video_path)

ret = True
# read frames
while ret:
    ret, frame = cap.read()

    if ret:
        # detect objects
        # track objects
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # plot results
        # cv2.rectangle
        # cv2.putText
        if results[0].boxes.id is not None:
            print(results[0].boxes.id.int().cpu().tolist())

        frame_ = results[0].plot()

        # visualize
        cv2.imshow("frame", frame_)
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break
