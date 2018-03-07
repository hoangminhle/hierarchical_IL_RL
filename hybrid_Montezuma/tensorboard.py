# Hoang M. Le
# California Institute of Technology
# hmle@caltech.edu
# (thanks to script from Malmo Project - Microsoft Corporation)
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
