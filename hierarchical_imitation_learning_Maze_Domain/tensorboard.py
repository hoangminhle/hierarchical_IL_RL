# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import absolute_import

import six
import tensorflow as tf
from tensorflow.core.framework.summary_pb2 import Summary

from visualizer import BaseVisualizer


class TensorboardVisualizer(BaseVisualizer):
    """
    Visualize the generated results in Tensorboard
    """

    def __init__(self):
        super(TensorboardVisualizer, self).__init__()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.01)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        self._train_writer = None

    def initialize(self, logdir, model, converter=None):
        assert logdir is not None, "logdir cannot be None"
        assert isinstance(logdir, six.string_types), "logdir should be a string"

        if converter is not None:
            assert isinstance(converter, TensorflowConverter), \
                        "converter should derive from TensorflowConverter"
            converter.convert(model, self._session.graph)

        self._train_writer = tf.summary.FileWriter(logdir=logdir,
                                                   graph=self._session.graph,
                                                   flush_secs=30)

    def add_entry(self, index, tag, value, **kwargs):
        if "image" in kwargs and value is not None:
            image_string = tf.image.encode_jpeg(value, optimize_size=True, quality=80)
            summary_value = Summary.Image(width=value.shape[1],
                                          height=value.shape[0],
                                          colorspace=value.shape[2],
                                          encoded_image_string=image_string)
        else:
            summary_value = Summary.Value(tag=tag, simple_value=value)

        if summary_value is not None:
            entry = Summary(value=[summary_value])
            self._train_writer.add_summary(entry, index)

    def close(self):
        if self._train_writer is not None:
            self._train_writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TensorflowConverter(object):
    def convert(self, network, graph):
        raise NotImplementedError()
