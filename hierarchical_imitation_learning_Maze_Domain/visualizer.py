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

from os import path


class Visualizable(object):
    def __init__(self, visualizer=None):
        if visualizer is not None:
            assert isinstance(visualizer, BaseVisualizer), "visualizer should derive from BaseVisualizer"

        self._visualizer = visualizer

    def visualize(self, index, tag, value, **kwargs):
        if self._visualizer is not None:
            self._visualizer << (index, tag, value, kwargs)

    @property
    def can_visualize(self):
        return self._visualizer is not None


class BaseVisualizer(object):
    """ Provide a unified interface for observing the training progress """

    def add_entry(self, index, key, result, **kwargs):
        raise NotImplementedError()

    def __lshift__(self, other):
        if isinstance(other, tuple):
            if len(other) >= 3:
                self.add_entry(other[0], str(other[1]), other[2])
            else:
                raise ValueError("Provided tuple should be of the form (key, value)")
        else:
            raise ValueError("Trying to use stream operator without a tuple (key, value)")


class EmptyVisualizer(BaseVisualizer):
    """ A boilerplate visualizer that does nothing """

    def add_entry(self, index, key, result, **kwargs):
        pass


class ConsoleVisualizer(BaseVisualizer):
    """ Print visualization to stdout as:
        key -> value
    """
    CONSOLE_DEFAULT_FORMAT = "[%s] %d : %s -> %.3f"

    def __init__(self, format=None, prefix=None):
        self._format = format or ConsoleVisualizer.CONSOLE_DEFAULT_FORMAT
        self._prefix = prefix or '-'

    def add_entry(self, index, key, result, **kwargs):
        print(self._format % (self._prefix, index, key, result))


class CsvVisualizer(BaseVisualizer):
    """ Write data to file. The following formats are supported: CSV, JSON, Excel. """
    def __init__(self, output_file, override=False):
        if path.exists(output_file) and not override:
            raise Exception('%s already exists and override is False' % output_file)

        super(CsvVisualizer, self).__init__()
        self._file = output_file
        self._data = {}

    def add_entry(self, index, key, result, **kwargs):
        if key in self._data[index]:
            print('Warning: Found previous value for %s in visualizer' % key)

        self._data[index].update({key: result})

    def close(self, format='csv'):
        import pandas as pd

        if format == 'csv':
            pd.DataFrame.from_dict(self._data, orient='index').to_csv(self._file)
        elif format == 'json':
            pd.DataFrame.from_dict(self._data, orient='index').to_json(self._file)
        else:
            writer = pd.ExcelWriter(self._file)
            pd.DataFrame.from_dict(self._data, orient='index').to_excel(writer)
            writer.save()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return self
