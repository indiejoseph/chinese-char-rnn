from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope

def adaptive_softmax_loss(inputs,
                          labels,
                          cutoff,
                          project_factor=4,
                          initializer=None,
                          name=None):
  """Computes and returns the adaptive softmax loss (a improvement of
  hierarchical softmax).

  See [Efficient softmax approximation for GPUs](https://arxiv.org/pdf/1609.04309v2.pdf).

  This is a faster way to train a softmax classifier over a huge number of
  classes, and can be used for **both training and prediction**. For example, it
  can be used for training a Language Model with a very huge vocabulary, and
  the trained languaed model can be used in speech recognition, text generation,
  and machine translation very efficiently.

  Args:
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
      activations of the input network.
    labels: `Tensor` of shape `[d_0, d_1, ..., d_{r-2}]` and dtype `int32` or
      `int64`. Each entry in `labels` must be an index in `[0, num_classes)`.
    cutoff: A list indicating the limits of the different clusters.
    project_factor: A floating point value greater or equal to 1.0. The projection
      factor between two neighboring clusters.
    initializer: Initializer for adaptive softmax variables (optional).
    name: A name for the operation (optional).
  Returns:
    loss: A `batch_size` 1-D tensor of the adaptive softmax cross entropy loss.
    training_losses: A list of 1-D tensors of adaptive softmax loss for each
      cluster, which can be used for calculating the gradients and back
      propagation when training.
  """
  input_dim = int(inputs.get_shape()[1])
  sample_num = int(inputs.get_shape()[0])
  cluster_num = len(cutoff) - 1
  with ops.name_scope(name, "AdaptiveSoftmax"):
    if initializer is None:
      stdv = math.sqrt(1. / input_dim)
      initializer = init_ops.random_uniform_initializer(-stdv * 0.8, stdv * 0.8)

    head_dim = cutoff[0] + cluster_num
    head_w = variable_scope.get_variable("adaptive_softmax_head_w",
                             [input_dim, head_dim], initializer=initializer)

    tail_project_factor = project_factor
    tail_w = []
    for i in range(cluster_num):
      project_dim = max(1, input_dim // tail_project_factor)
      tail_dim = cutoff[i + 1] - cutoff[i]
      tail_w.append([
        variable_scope.get_variable("adaptive_softmax_tail{}_proj_w".format(i+1),
                        [input_dim, project_dim], initializer=initializer),
        variable_scope.get_variable("adaptive_softmax_tail{}_w".format(i+1),
                        [project_dim, tail_dim], initializer=initializer)
      ])
      tail_project_factor *= project_factor

    # Get tail masks and update head labels
    training_losses = []
    loss = array_ops.zeros([sample_num], dtype=dtypes.float32)
    head_labels = labels
    for i in range(cluster_num):
      mask = math_ops.logical_and(math_ops.greater_equal(labels, cutoff[i]),
                                  math_ops.less(labels, cutoff[i + 1]))

      # Update head labels
      head_labels = array_ops.where(mask, array_ops.constant([cutoff[0] + i] *
                            sample_num), head_labels)

      # Compute tail loss
      tail_inputs = array_ops.boolean_mask(inputs, mask)
      tail_logits = math_ops.matmul(math_ops.matmul(tail_inputs, tail_w[i][0]),
                                    tail_w[i][1])
      tail_labels = array_ops.boolean_mask(labels - cutoff[i], mask)
      tail_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=tail_logits,
                                                              labels=tail_labels)
      training_losses.append(tail_loss)
      aligned_tail_loss = sparse_tensor.SparseTensor(
        array_ops.squeeze(array_ops.where(mask)), tail_loss, [sample_num])
      loss += sparse_ops.sparse_tensor_to_dense(aligned_tail_loss)

    # Compute head loss
    head_logits = math_ops.matmul(inputs, head_w)
    head_loss = nn.sparse_softmax_cross_entropy_with_logits(logits=head_logits,
                                                            labels=head_labels)
    loss += head_loss
    training_losses.append(head_loss)

    return loss, training_losses
