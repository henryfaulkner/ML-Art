import os
from cv2 import cv2
import matplotlib.pyplot as plt
from MachineLearningProcessor.VisualizeImages import VisualizeImages
from PIL import Image


class GenerateVideo:
    def ResizeImages(self):
        '''Currently unnecessary, using like images'''
        return

    def GenerateVideo(self, imgs, num_iterations, video_file_name):
        '''
            Arguements:
                imgs: list of numpy.ndarray objects of all iterations
            '''
        try:
            vs = VisualizeImages()

            video_name = "../Videos/" + video_file_name + ".avi"
            print("img shape length: ", str(len(imgs[0].shape)))
            print("img tuple: ", str(imgs[0].shape))
            print(type(imgs[0]))
            height, width, channel = imgs[0].shape  # numpy.ndarray
            fps = num_iterations/10

            video = cv2.VideoWriter(
                video_name, 0, fps, (width, height))

            # Appending the images to the video one by one
            for img in imgs:
                # convert RGB image to BGR for openCV
                frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                video.write(frame)

            # Deallocating memories taken for window creation
            cv2.destroyAllWindows()
            video.release()  # releasing the video generated
        except:
            print("shit")

        return
