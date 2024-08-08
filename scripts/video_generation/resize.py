import cv2


def resize_video(input_path, output_path, width, height):
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        resized_frame = cv2.resize(frame, (width, height))
        out.write(resized_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    scenes = [1, 2, 3, 4, 5, 6]
    outputs = ["real"]

    for scene in scenes:
        for output in outputs:
            input_path = f"/home/ge42top/Desktop/output/video_{scene}_{output}.avi"
            output_path = f"/home/ge42top/Desktop/output/1_video_{scene}_{output}.avi"
            width = 1024  # Desired width
            height = 512  # Desired height

            resize_video(input_path, output_path, width, height)
