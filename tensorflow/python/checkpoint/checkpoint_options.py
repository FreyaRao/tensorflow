# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Options for saving Checkpoints."""

from tensorflow.python.util.tf_export import tf_export


@tf_export("train.CheckpointOptions")
class CheckpointOptions(object):
  """Options for constructing a Checkpoint.

  Used as the `options` argument to either `tf.train.Checkpoint.save()` or
  `tf.train.Checkpoint.restore()` methods to adjust how variables are
  saved/restored.

  Example: Run IO ops on "localhost" while saving a checkpoint:

  ```
  step = tf.Variable(0, name="step")
  checkpoint = tf.train.Checkpoint(step=step)
  options = tf.train.CheckpointOptions(experimental_io_device="/job:localhost")
  checkpoint.save("/tmp/ckpt", options=options)
  ```
  """

  # Define object attributes in __slots__ for improved memory and performance.
  __slots__ = ("experimental_io_device", "experimental_enable_async_checkpoint", "enable_nebula")

  def __init__(self, experimental_io_device=None,
               experimental_enable_async_checkpoint=False,
               enable_nebula=False):
    """Creates an object that stores options for a Checkpoint.

    Args:
      experimental_io_device: string. Applies in a distributed setting.
        Tensorflow device to use to access the filesystem. If `None` (default)
        then for each variable the filesystem is accessed from the CPU:0 device
        of the host where that variable is assigned. If specified, the
        filesystem is instead accessed from that device for all variables.

        This is for example useful if you want to save to a local directory,
        such as "/tmp" when running in a distributed setting. In that case pass
        a device for the host where the "/tmp" directory is accessible.

      experimental_enable_async_checkpoint: bool Type. Indicates whether async
        checkpoint is enabled. Default is False, i.e., no async checkpoint.

        Async checkpoint moves the checkpoint file writing off the main thread,
        so that the model can continue to train while the checkpoing file
        writing runs in the background. Async checkpoint reduces TPU device idle
        cycles and speeds up model training process, while memory consumption
        may increase.
    """
    self.experimental_io_device = experimental_io_device
    self.experimental_enable_async_checkpoint = experimental_enable_async_checkpoint
    self.enable_nebula = enable_nebula
