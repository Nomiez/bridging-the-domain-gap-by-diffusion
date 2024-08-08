import cv2
import os

scenes = [1, 2, 3, 4, 5, 6]
outputs = ["real"]

for scene in scenes:
    for output in outputs:
        image_folder = f"/home/ge42top/Desktop/output/input_8_100/Scene{scene}/output_{output}/"
        video_name = f"video_{scene}_{output}.avi"

        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images = sorted(images, key=lambda name: int(name.split(".")[0]))
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        print(height, width)

        video = cv2.VideoWriter(video_name, 0, 10, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))

        cv2.destroyAllWindows()
        video.release()
