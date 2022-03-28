from MachineLearningProcessor.VisualizeImages import VisualizeImages
from MachineLearningProcessor.PrepareData import PrepareData
from MachineLearningProcessor.GradientDescent import GradientDescent
from MetAPI.RESTCalls import RESTCalls
from helpers.FileStructures import FileStructures
from keras.preprocessing import image
from keras import models
import tensorflow as tf
import functools
import time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
mpl.rcParams['figure.figsize'] = (10, 10)
mpl.rcParams['axes.grid'] = False


def main():
    vi = VisualizeImages()
    fs = FileStructures()
    numIterations = input('How many processing iterations?')
    folder_and_file_name = 'temp'  # fs.CreateFileInput()
    fs.CreateResultFolder(folder_and_file_name)
    content_style_tuple = fs.AccessImageInput(folder_and_file_name)

    content_path = content_style_tuple[0]
    style_path = content_style_tuple[1]

    # Content layer where will pull our feature maps
    content_layers = ['block5_conv2']

    # Style layer we are interested in
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1'
                    ]

    best_processed_art, best_loss = run_style_transfer(
        content_path, style_path, content_layers, style_layers, num_iterations=numIterations)

    folder_and_file_name = fs.RenameResultFolderName(
        folder_and_file_name, content_style_tuple)
    content_path = "../Images/" + \
        folder_and_file_name + "/" + content_style_tuple[2]['FileName']
    style_path = "../Images/" + \
        folder_and_file_name + "/" + content_style_tuple[3]['FileName']
    vi.show_results(best_processed_art, content_path, style_path)
    fs.SaveResultFile(best_processed_art, folder_and_file_name, numIterations)


def get_model(content_layers, style_layers):
    """ Creates our model with access to intermediate layers. 

    This function will load the VGG19 model and access the intermediate layers. 
    These layers will then be used to create a new model that will take input image
    and return the outputs from these intermediate layers from the VGG model. 

    Returns:
        returns a keras model that takes image inputs and outputs the style and 
        content intermediate layers. 
    """
    # Load our model. We load pretrained VGG, trained on imagenet data
    vgg = tf.keras.applications.vgg19.VGG19(
        include_top=False, weights='imagenet')
    vgg.trainable = False
    # Get output layers corresponding to style and content layers
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    model_outputs = style_outputs + content_outputs
    # Build model
    return models.Model(vgg.input, model_outputs)


def get_feature_representations(model, content_path, style_path, content_layers, style_layers):
    """Helper function to compute our content and style feature representations.

    This function will simply load and preprocess both the content and style 
    images from their path. Then it will feed them through the network to obtain
    the outputs of the intermediate layers. 

    Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image

    Returns:
    returns the style features and the content features. 
    """
    vi = VisualizeImages()
    pd = PrepareData()

    # Load our images in
    content_image = pd.load_and_process_img(vi, content_path)
    style_image = pd.load_and_process_img(vi, style_path)

    # batch compute content and style features
    style_outputs = model(style_image)
    content_outputs = model(content_image)

    # Get the style and content feature representations from our model
    style_features = [style_layer[0]
                      for style_layer in style_outputs[:len(style_layers)]]
    content_features = [content_layer[0]
                        for content_layer in content_outputs[len(style_layers):]]
    return style_features, content_features


def run_style_transfer(content_path,
                       style_path,
                       content_layers,
                       style_layers,
                       num_iterations=1000,
                       content_weight=1e3,
                       style_weight=1e-2):
    gd = GradientDescent()
    pd = PrepareData()
    vi = VisualizeImages()

    # Not training the model
    model = get_model(content_layers, style_layers)
    for layer in model.layers:
        layer.trainable = False

    # Get the style and content feature representations (from our specified intermediate layers)
    style_features, content_features = get_feature_representations(
        model, content_path, style_path, content_layers, style_layers)
    gram_style_features = [gd.gram_matrix(style_feature)
                           for style_feature in style_features]

    # Set initial image
    init_image = pd.load_and_process_img(vi, content_path)
    init_image = tf.Variable(init_image, dtype=tf.float32)
    # Create our optimizer
    opt = tf.optimizers.Adam(
        learning_rate=5, beta_1=0.99, epsilon=1e-1)

    # For displaying intermediate images
    iter_count = 1

    # Store our best result
    best_loss, best_img = float('inf'), None

    # Create a nice config
    loss_weights = (style_weight, content_weight)
    cfg = {
        'model': model,
        'loss_weights': loss_weights,
        'init_image': init_image,
        'gram_style_features': gram_style_features,
        'content_features': content_features,
        'num_content_layers': len(content_layers),
        'num_style_layers': len(style_layers)
    }

    # For displaying
    num_rows = 2
    num_cols = 5
    display_interval = num_iterations/(num_rows*num_cols)
    start_time = time.time()
    global_start = time.time()

    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    imgs = []
    for i in range(num_iterations):
        grads, all_loss = gd.calc_gradients(cfg)
        loss, style_score, content_score = all_loss
        opt.apply_gradients([(grads, init_image)])
        clipped = tf.clip_by_value(init_image, min_vals, max_vals)
        init_image.assign(clipped)
        end_time = time.time()

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = pd.deprocess_img(init_image.numpy())

        if i % display_interval == 0:
            start_time = time.time()
            plot_img = init_image.numpy()
            plot_img = pd.deprocess_img(plot_img)
            print('Iteration: {}'.format(i))
            print('Total loss: {:.4e}, '
                  'style loss: {:.4e}, '
                  'content loss: {:.4e}, '
                  'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
    print('Total time: {:.4f}s'.format(time.time() - global_start))
    for i, img in enumerate(imgs):
        plt.subplot(num_rows, num_cols, i+1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])

    return best_img, best_loss


if __name__ == '__main__':
    main()
