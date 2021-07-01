from pathlib import Path
import skimage
from skimage import io
from skimage.transform import resize
import numpy as np
from sklearn.utils import Bunch

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

    # return in the exact same format as the built-in datasets
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)

dataset = load_image_files("/home/joshuavanstaden/Datasets/Oxythyrea_images/training")
print(dataset.target_names)
