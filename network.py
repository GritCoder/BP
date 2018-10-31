"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
import mnist_loader

class Network(object):

    def __init__(self, sizes):#初始化
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)#假设这个网络是三层的
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]#生成2X1 3X1的biases列表
        self.weights = [np.random.randn(y, x)#生成1X2 2X3的weights列表  注意一下，这都不是随意生成的，都是为了方便后面的就诊计算而生成的
                        for x, y in zip(sizes[:-1], sizes[1:])]#注意切片列表都不包括端点值

    def feedforward(self, a):#前馈网络
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)#前馈网络，计算sigmoid函数后的a的输出值
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):#随机梯度下降法学习参数
        """Train the neural network usingbat mini-ch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):#每一次迭代
            #random.shuffle(training_data)#将序列的所有元素随机排序
            mini_batches = [
                training_data[k:k+mini_batch_size]#返回一个列表切片，然后再把列表切片添加到一个新的列表中
                for k in range(0, n, mini_batch_size)]#设置了一个步长  最为最小的训练规模
            for mini_batch in mini_batches:#然后对列表中的每一个切片
                self.update_mini_batch(mini_batch, eta)#传入数据和学习率，进行梯度下降法的学习
            if test_data:
                print ("Epoch {0}: {1} / {2}").format(
                    j, self.evaluate(test_data), n_test)
            else:
                print ("Epoch {0} complete").format(j)

    def update_mini_batch(self, mini_batch, eta):#具体的学习算法实现
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]#初始化b and w 的偏微分为0
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for item in mini_batch:#对于每一组数据，执行下列操作
            delta_nabla_b, delta_nabla_w = self.backprop(item[0], item[1])#反向传播的计算 b w 的偏微分
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw  #更新w  #现在明白了为什么要除数据集的长度，因为用的是均值误差
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb   #更新b
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):#利用反向传播法计算b w的偏微分，代表损失函数的梯度，因为损失函数最后就剩b w的函数
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]#初始化b w 的偏微分
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x #初始激励值
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b#计算激励值输出
            zs.append(z)#激励输出列表(没有进行sigmoid转化)
            activation = sigmoid(z) #sigmoid函数转一下
            activations.append(activation)#将转之后的激励输出加入列表
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1]) #计算最后一层的delta = 该层的偏差 * 对应的该层激励输出对应的偏导数
        nabla_b[-1] = delta  #往前以此类推，计算方式也都是类似的， 用delta的值更新最后一层的b
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())# 最后一层的w = delta * 对应的输入激励值（即倒数第二层的输出）
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)# 这里的计算核心要理解：前一层的输出是后一层的输入
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp#偏差 * 该层对应的输出激励 含义即为误差的偏导数
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)#返回每一层的更新过的b w偏导数  注意都是矩阵运算，返回的都是矢量

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)#注意百度查一下argmax函数的作用
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y): #计算每一层的误差E
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):#可以理解为正则化函数
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):#对sigmoid函数求偏导  根据推导公式中的需要得出
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))#通俗点来说就是偏微分
def main():#测试类方法
    training_data, validation_data, test_data = mnist_loader.load_data()#获取数据集
    test = Network([2,3,1]) #创建一个类的实例 但是数据集是784维度的，传入一个784的列表不现实，这个demo目前跑不了，理解思路即可
    test.SGD(training_data,20,5,0.01,test_data)
if __name__ == "__main__":
    main()