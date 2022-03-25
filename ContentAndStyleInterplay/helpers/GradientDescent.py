from keras.preprocessing import image
import tensorflow as tf


class GradientDescent:

    def calc_content_loss(self, base_content, layer_target):
        ''' Calculate the distance of content from the output image
            to the base input image.
            Add content losses at each desired layer.
        '''
        return tf.reduce_mean(tf.square(base_content - layer_target))

    def calc_style_loss(self, base_style, gram_target):
        ''' Compare Gram matricies of the output image to the base
            input image.
            Expects two images of dimension h, w, c
        '''
        # height, width, num filters of each layer
        # We scale the loss at a given layer by the size of the feature map and the number of filters
        height, width, channels = base_style.get_shape().as_list()
        gram_style = self.gram_matrix(base_style)

        # / (4. * (channels ** 2) * (width * height) ** 2)
        return tf.reduce_mean(tf.square(gram_style - gram_target))

    def gram_matrix(self, input_tensor):
        # We make the image channels first
        channels = int(input_tensor.shape[-1])
        a = tf.reshape(input_tensor, [-1, channels])
        n = tf.shape(a)[0]
        gram = tf.matmul(a, a, transpose_a=True)
        return gram / tf.cast(n, tf.float32)

    def calc_loss(self, model, loss_weights, init_image,  content_features, gram_style_features, num_content_layers, num_style_layers):
        """This function will compute the loss total loss.

        Arguments:
            model: The model that will give us access to the intermediate layers
            loss_weights: The weights of each contribution of each loss function. 
            (style weight, content weight, and total variation weight)
            init_image: Our initial base image. This image is what we are updating with 
            our optimization process. We apply the gradients wrt the loss we are 
            calculating to this image.
            content_features: Precomputed outputs from defined content layers of 
            interest.
            gram_style_features: Precomputed gram matrices corresponding to the 
            defined style layers of interest.

        Returns:
            returns the total loss, style loss, content loss, and total variational loss
        """
        style_weight, content_weight = loss_weights

        # Feed our init image through our model. This will give us the content and
        # style representations at our desired layers. Since we're using eager
        # our model is callable just like any other function!
        model_outputs = model(init_image)

        style_output_features = model_outputs[:num_style_layers]
        content_output_features = model_outputs[num_style_layers:]

        style_score = 0
        content_score = 0

        # Accumulate style losses from all layers
        # Here, we equally weight each contribution of each loss layer
        weight_per_style_layer = 1.0 / float(num_style_layers)
        for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * \
                self.calc_style_loss(comb_style[0], target_style)

        # Accumulate content losses from all layers
        weight_per_content_layer = 1.0 / float(num_content_layers)
        for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer * \
                self.calc_content_loss(comb_content[0], target_content)

        style_score *= style_weight
        content_score *= content_weight

        # Get total loss
        loss = style_score + content_score
        return loss, style_score, content_score

    def calc_gradients(self, cfg):
        with tf.GradientTape() as tape:
            all_loss = self.calc_loss(**cfg)
        # Compute gradients wrt input image
        total_loss = all_loss[0]
        return tape.gradient(total_loss, cfg['init_image']), all_loss
