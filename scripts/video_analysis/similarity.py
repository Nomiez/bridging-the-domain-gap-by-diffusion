from image_similarity_measures.evaluate import evaluation
import os
from pathlib import Path


def list_files(directory):
    return [
        f"{directory}/{f}"
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]


scenes = [1, 2, 3, 4, 5, 6]
outputs = ["real"]
file_path = Path("/home/ge42top/Desktop/output/output_real.txt")

string = ""

for scene in scenes:
    for output in outputs:
        # Example usage:
        directory = f"/home/ge42top/Desktop/output/input_8_100/Scene{scene}/output_{output}/"
        files = list(filter(lambda x: ".png" in x, list_files(directory)))
        files = sorted(files, key=lambda name: int(name.split(".")[0].split("/")[-1]))
        res = []
        for i in range(1, len(files)):
            print(files[i - 1])
            print(files[i])
            res.append(
                evaluation(org_img_path=files[i - 1], pred_img_path=files[i], metrics=["rmse"])[
                    "rmse"
                ]
            )
        string += f"{scene}, {output}: {str(res)}" + "\n"
    string += "\n"


file_path.write_text(string)
