from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import six
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow_gan as tfgan
import tensorflow_hub as hub


def preprocess(videos, target_resolution):
  """Runs some preprocessing on the videos for I3D model.

  Args:
    videos: <T>[batch_size, num_frames, height, width, depth] The videos to be
      preprocessed. We don't care about the specific dtype of the videos, it can
      be anything that tf.image.resize_bilinear accepts. Values are expected to
      be in the range 0-255.
    target_resolution: (width, height): target video resolution

  Returns:
    videos: <float32>[batch_size, num_frames, height, width, depth]
  """
  videos_shape = videos.shape.as_list()
  all_frames = tf.reshape(videos, [-1] + videos_shape[-3:])
  resized_videos = tf.image.resize_bilinear(all_frames, size=target_resolution)
  target_shape = [videos_shape[0], -1] + list(target_resolution) + [3]
  output_videos = tf.reshape(resized_videos, target_shape)
  scaled_videos = 2. * tf.cast(output_videos, tf.float32) / 255. - 1
  return scaled_videos


def _is_in_graph(tensor_name):
  """Checks whether a given tensor does exists in the graph."""
  try:
    tf.get_default_graph().get_tensor_by_name(tensor_name)
  except KeyError:
    return False
  return True


def create_id3_embedding(videos):
    """Embeds the given videos using the Inflated 3D Convolution network.

    Downloads the graph of the I3D from tf.hub and adds it to the graph on the
    first call.

    Args:
        videos: <float32>[batch_size, num_frames, height=224, width=224, depth=3].
        Expected range is [-1, 1].

    Returns:
        embedding: <float32>[batch_size, embedding_size]. embedding_size depends
                   on the model used.

    Raises:
        ValueError: when a provided embedding_layer is not supported.
    """

    batch_size = 16
    module_spec = "https://tfhub.dev/deepmind/i3d-kinetics-400/1"

    # Making sure that we import the graph separately for
    # each different input video tensor.
    module_name = "fvd_kinetics-400_id3_module_" + six.ensure_str(
        videos.name).replace(":", "_")

    assert_ops = [
        tf.Assert(
            tf.reduce_max(videos) <= 1.001,
            ["max value in frame is > 1", videos]),
        tf.Assert(
            tf.reduce_min(videos) >= -1.001,
            ["min value in frame is < -1", videos]),
        tf.assert_equal(
            tf.shape(videos)[0],
            batch_size, ["invalid frame batch size: ",
                        tf.shape(videos)],
            summarize=6),
    ]
    with tf.control_dependencies(assert_ops):
        videos = tf.identity(videos)

    module_scope = "%s_apply_default/" % module_name

    # Directly use hub.KerasLayer with the module_spec URL
    i3d_layer = hub.KerasLayer(module_spec, trainable=False)

    # Apply the model to the videos
    tensor = i3d_layer(videos)
    return tensor


def calculate_fvd(real_activations, generated_activations):
    """Returns a scalar that contains the requested FVD.

    Args:
        real_activations: <float32>[num_samples, embedding_size]
        generated_activations: <float32>[num_samples, embedding_size]

    Returns:
        A scalar that contains the requested FVD.
    """
    with tf.Session() as sess:
        # 初始化由 hub/KerasLayer 等创建的变量和表
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(tf.tables_initializer())

        fvd = tfgan.eval.frechet_classifier_distance_from_activations(
            real_activations, generated_activations)
        return sess.run(fvd)