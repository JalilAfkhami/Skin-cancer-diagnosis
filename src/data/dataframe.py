import pandas as pd
import os

def export_label(path):
    df = pd.read_csv(path, sep=" ", header=None, on_bad_lines='skip')
    return df[0].values[0]


def make_csv(path):
    image_folder_root = path + "/images"
    label_folder_root = path + "/labels"
    name = []
    image_path = []
    label = []
    valid_images = [".jpg",".bmp",".png",".jpeg"]
    for f in os.listdir(image_folder_root):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        image_path.append(f)
        name.append(os.path.splitext(f)[0])
        
    for f in name:
        label.append(
            export_label(label_folder_root + "/" + f +'.txt')
                )
        
    df = pd.DataFrame(
        {'image_path': image_path,
         "label": label}
    )
    
    return df