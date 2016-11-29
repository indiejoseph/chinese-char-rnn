import tensorflow as tf
import numpy as np
from base import Model
import ops

class ByteNet(Model):
  def __init__(self,
    n_source_quant=1000, n_target_quant=1000,
    residual_channels=256, batch_size=100, seq_length=200,
    encoder_filter_width=3, decoder_filter_width=3,
    encoder_dilations="1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16",
    decoder_dilations="1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16,1,2,4,8,16",
    decay_rate=0.95, learning_rate=0.001, learning_rate_step=1000, grad_norm=5.0,
    checkpoint_dir="checkpoint", dataset_name="wiki"
  ):
    """
    n_source_quant : quantization channels of source text
    n_target_quant : quantization channels of target text
    residual_channels : number of channels in internal blocks
    batch_size : Batch Size
    seq_length : Text Sample Length
    encoder_filter_width : Encoder Filter Width
    decoder_filter_width : Decoder Filter Width
    encoder_dilations : Dilation Factor for decoder layers (list)
    decoder_dilations : Dilation Factor for decoder layers (list)
    """
    Model.__init__(self)

    self.n_source_quant = n_source_quant
    self.n_target_quant = n_target_quant
    self.residual_channels = residual_channels
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.encoder_filter_width = encoder_filter_width
    self.decoder_filter_width = decoder_filter_width
    self.encoder_dilations = [int(d) for d in encoder_dilations.split(",")]
    self.decoder_dilations = [int(d) for d in decoder_dilations.split(",")]
    self.checkpoint_dir = checkpoint_dir
    self.decay_rate = decay_rate
    self.learning_rate = learning_rate
    self.learning_rate_step = learning_rate_step
    self.grad_norm = grad_norm
    self.dataset_name = dataset_name

    self.w_source_embedding = tf.get_variable("w_source_embedding",
      [self.n_source_quant, 2*self.residual_channels]
      initializer=tf.truncated_normal_initializer(stddev=0.02)
    )

    # TO BE CONCATENATED WITH THE ENCODER EMBEDDING
    self.w_target_embedding = tf.get_variable("w_target_embedding",
      [self.n_target_quant, self.residual_channels]
      initializer=tf.truncated_normal_initializer(stddev=0.02)
    )

    self.sentence = tf.placeholder("int32", [self.batch_size, self.seq_length], name="sentence")
    self.targets = tf.placeholder("int32", [self.batch_size, self.seq_length], name="sentence")

    with tf.device("/cpu:0"):
      source_embedding = tf.nn.embedding_lookup(self.w_source_embedding, self.sentence, name="source_embedding")

    outputs = self.decoder(source_embedding)

    # Loss
    self.logits = tf.reshape(outputs, [-1, self.n_target_quant])
    self.prediction = tf.argmax(self.logits, 1)
    self.probs = tf.nn.softmax(self.logits)

    labels = tf.reshape(self.targets, [-1])
    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, labels)
    self.cost = tf.reduce_mean(self.loss)

    self.global_step = tf.Variable(0, name='global_step', trainable=False)

    tvars = tf.trainable_variables()
    lr = tf.train.exponential_decay(self.learning_rate, self.global_step, self.learning_rate_step,
                                    self.decay_rate, staircase=True)

    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), grad_norm)
    optimizer = tf.train.AdamOptimizer(lr)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars),global_step=self.global_step)

  def decode_layer(self, input_, dilation, layer_no):
    relu1 = tf.nn.relu(input_, name="dec_relu1_layer{}".format(layer_no))
    conv1 = ops.conv1d(relu1, self.residual_channels, name="dec_conv1d_1_layer{}".format(layer_no))

    relu2 = tf.nn.relu(conv1, name="enc_relu2_layer{}".format(layer_no))
    dilated_conv = ops.dilated_conv1d(relu2, self.residual_channels,
      dilation, self.decoder_filter_width,
      causal = True,
      name = "dec_dilated_conv_laye{}".format(layer_no)
      )

    relu3 = tf.nn.relu(dilated_conv, name="dec_relu3_layer{}".format(layer_no))
    conv2 = ops.conv1d(relu3, 2 * self.residual_channels, name="dec_conv1d_2_layer{}".format(layer_no))

    return input_ + conv2

  def decoder(self, input_, encoder_embedding = None):
    curr_input = input_
    if encoder_embedding != None:
      # CONDITION WITH ENCODER EMBEDDING FOR THE TRANSLATION MODEL
      curr_input = tf.concat(2, [input_, encoder_embedding])
      print "Decoder Input", curr_input

    for layer_no, dilation in enumerate(self.decoder_dilations):
      layer_output = self.decode_layer(curr_input, dilation, layer_no)
      curr_input = layer_output

    layer_output = tf.nn.relu(layer_output)

    processed_output = ops.conv1d(layer_output,
      self.n_target_quant,
      name="decoder_post_processing")

    return processed_output

  def encode_layer(self, input_, dilation, layer_no):
    relu1 = tf.nn.relu(input_, name="enc_relu1_layer{}".format(layer_no))
    conv1 = ops.conv1d(relu1, self.residual_channels, name="enc_conv1d_1_layer{}".format(layer_no))
    conv1 = tf.mul(conv1, self.source_masked_d)
    relu2 = tf.nn.relu(conv1, name="enc_relu2_layer{}".format(layer_no))
    dilated_conv = ops.dilated_conv1d(relu2, self.residual_channels,
      dilation, self.encoder_filter_width,
      causal = False,
      name = "enc_dilated_conv_layer{}".format(layer_no)
      )
    dilated_conv = tf.mul(dilated_conv, self.source_masked_d)
    relu3 = tf.nn.relu(dilated_conv, name="enc_relu3_layer{}".format(layer_no))
    conv2 = ops.conv1d(relu3, 2 * self.residual_channels, name="enc_conv1d_2_layer{}".format(layer_no))
    return input_ + conv2

  def encoder(self, input_):
    curr_input = input_
    for layer_no, dilation in enumerate(self.self.encoder_dilations):
      layer_output = self.encode_layer(curr_input, dilation, layer_no)

      # ENCODE ONLY TILL THE INPUT LENGTH, conditioning should be 0 beyond that
      layer_output = tf.mul(layer_output, self.source_masked, name="layer_{}_output".format(layer_no))

      curr_input = layer_output

    # TO BE CONCATENATED WITH TARGET EMBEDDING
    processed_output = tf.nn.relu( ops.conv1d(tf.nn.relu(layer_output),
      self.residual_channels,
      name="encoder_post_processing") )

    processed_output = tf.mul(processed_output, self.source_masked_d, name="encoder_processed")

    return processed_output


if __name__ == "__main__":
  model = ByteNet()
