from pathlib import Path
from argparse import ArgumentParser
import skimage
from skimage import io
from skimage.transform import resize
import numpy as np
from sklearn.utils import Bunch
import pandas as pd

def directory_to_dataframe(container_path, dimension=(32, 32, 3)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]
    images = []
    flat_data = []
    target = []
    target_str = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = io.imread(file)
            img_resized = resize(img[:,:,:3], dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized) 
            images.append(img_resized)
            target.append(i)
            target_str.append(str(direc).split("/")[-1])
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    img_dict = {}
    for i in range(flat_data.shape[1]):
        for j in range(flat_data.shape[2]):
            for k in range(flat_data.shape[3]):
                img_dict["pix_%s_%s_%s" %(i,j,k)] = flat_data[:,i,j,k]
    img_dict["target"] = target
    img_dict["target_str"] = target_str
    return pd.DataFrame.from_dict(img_dict)


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument("--train", default="/home/joshuavanstaden/Datasets/Oxythyrea_images/training")
    args.add_argument("--test", default="/home/joshuavanstaden/Datasets/Oxythyrea_images/testing")
    args = args.parse_args()

    train = directory_to_dataframe(args.train)
    test = directory_to_dataframe(args.test)
    train.to_pickle("train.pkl")
    test.to_pickle("test.pkl")
    print("Saved Files")


