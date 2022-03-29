import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from keras.preprocessing import image as kp_image
import os


class VisualizeImages:
    def load_img(self, path_to_img):
        max_dim = 512
        scriptDir = os.path.dirname(__file__)
        impath = os.path.join(scriptDir, "../" + path_to_img)
        img = Image.open(impath)
        long = max(img.size)
        scale = max_dim/long
        img = img.resize(
            (round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)

        img = kp_image.img_to_array(img)

        # Need to broadcast the image array such that it has a batch dimension
        img = np.expand_dims(img, axis=0)
        return img

    def imshow(self, img, title=None):
        # Remove the batch dimension
        out = np.squeeze(img, axis=0)
        # Normalize for display
        out = out.astype('uint8')
        plt.imshow(out)
        if title is not None:
            plt.title(title)
        plt.imshow(out)

    def DisplayContentAndStyleImage(self, content_path, style_path):
        content = self.load_img(
            content_path).astype('uint8')
        style = self.load_img(
            style_path).astype('uint8')

        plt.subplot(1, 2, 1)
        self.imshow(content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.imshow(style, 'Style Image')
        plt.show()

    def show_results(self, best_img, content_path, style_path, show_large_final=True):
        plt.figure(figsize=(10, 5))
        content = self.load_img(content_path)
        style = self.load_img(style_path)

        plt.subplot(1, 2, 1)
        self.imshow(content, 'Content Image')

        plt.subplot(1, 2, 2)
        self.imshow(style, 'Style Image')

        if show_large_final:
            plt.figure(figsize=(10, 10))

            plt.imshow(best_img)
            plt.title('Output Image')
            plt.show()
