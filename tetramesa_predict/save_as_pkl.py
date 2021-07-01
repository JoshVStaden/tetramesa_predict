from pathlib import Path
import skimage
from skimage import io
from skimage.transform import resize
import numpy as np
from sklearn.utils import Bunch
import pandas as pd

def load_image_files(container_path, dimension=(30, 30, 3)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "Oxythrea Images"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)

    # print(flat_data.shape)
    # quit()
    img_dict = {}
    for i in range(flat_data.shape[1]):
        for j in range(flat_data.shape[2]):
            for k in range(flat_data.shape[3]):
                img_dict["pix_%s_%s_%s" %(i,j,k)] = flat_data[:,i,j,k]

    img_dict["target"] = target
    return pd.DataFrame.from_dict(img_dict)

dataset = load_image_files("/home/joshuavanstaden/Datasets/Oxythyrea_images/training")
dataset.to_pickle("Oxythyrea_dataframe.pkl")
print("Saved File")
