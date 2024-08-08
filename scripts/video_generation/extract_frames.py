import cv2
import os


def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"{frame_number:04d}.png")
        cv2.imwrite(frame_path, frame)

        frame_number += 1
        print(f"Extracted frame {frame_number}/{total_frames}")

        if frame_number > 300:
            break

    cap.release()
    print("Finished extracting frames.")


# Example usage
scenes = [1, 2, 3, 4, 5, 6]
for scene in scenes:
    video_path = f"/home/ge42top/Desktop/video/scene{scene}/video.mp4"
    output_folder = f"/home/ge42top/Desktop/video/scene{scene}/output_real"
    extract_frames(video_path, output_folder)
