"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle #用来序列化对象的一个标准库
import gzip #用来读取压缩文件的一个标准库

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = open('../data/mnist.pkl', 'rb')
    training_data, validation_data, test_data = pickle.load(f,encoding='bytes')#加载训练集、验证机、测试集
    f.close()
    return (training_data, validation_data, test_data)#以元组的形式返回

def load_data_wrapper():#对加载的数据做一些格式化的处理
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``.  ``x`` is atraining_data`` is a list containing 50,000
    2-tuples ``(x, y)`` 784-dimensional numpy.ndarray
    containing the input age. im ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
    #reshape()中x的含义是要进行重塑的数组，元组代表重塑后的形状（查看文档得知）
    tr_d, va_d, te_d = load_data() #心里要理解为什么只处理训练集、而对验证集和测试集不做任何处理。！
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]#高维的reshape有点懵逼 待日后再研究一下
    training_results = [vectorized_result(y) for y in tr_d[1]]#初始化结果集result
    training_data = zip(training_inputs, training_results) #打包训练集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]#验证集，注意不需要对Y进行特殊处理
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])#仍然不需要特殊处理
    return (training_data, validation_data, test_data)#返回处理后的训练集、验证集、测试集

def vectorized_result(j): #用来初始化训练集中的Y的
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
# if __name__ == "__main__":
#     training_data ,validation_data , test_data , = load_data_wrapper()
#     print(training_data)
#     print(validation_data)
#     print(test_data)