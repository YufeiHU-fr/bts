from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


if __name__=='__main__':
    img_rgb = Image.open('./visualization/rgb/munster_munster_000004_000019_leftImg8bit.png').convert('RGB')
    img_camp = Image.open('./visualization/cmap/munster_munster_000004_000019_leftImg8bit.png').convert('L')

    plt.Figure(figsize=(12,8))
    plt.subplots(121)
    plt.title('original image')
    plt.axis('off')
    plt.imshow(img_rgb)
    plt.show()