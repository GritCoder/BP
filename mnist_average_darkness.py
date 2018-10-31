"""
mnist_average_darkness  #此种方式用的是基本的线性分类方法的结果
~~~~~~~~~~~~~~~~~~~~~~
#大意就是根据像素点来判断数字，函数返回的是判断的数据，以及在训练集中该数字的平均像素点
A naive classifier for recognizing handwritten digits from the MNIST
data set.  The program classifies digits based on how dark they are
--- the idea is that digits like "1" tend to be less dark than digits
like "8", simply because the latter has a more complex shape.  When
shown an image the classifier returns whichever digit in the training
data had the closest average darkness.
# 程序分为两步：1、训练分类器 2、在mnist测试集上应用分类器查看效果
The program works in two steps: first it trains the classifier, and
then it applies the classifier to the MNIST test data to see how many
digits are correctly classified.
#该种方法不是最好的，但是仍然可以展示一下具体的思想
Needless to say, this isn't a very good way of recognizing handwritten
digits!  Still, it's useful to show what sort of performance we get
from naive ideas."""

#### Libraries
# Standard library
from collections import defaultdict

# My libraries
import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data()
    # training phase: compute the average darknesses for each digit,
    # based on the training data
    avgs = avg_darknesses(training_data)
    # testing phase: see how many of the test images are classified
    # correctly
    num_correct = sum(int(guess_digit(image, avgs) == digit)
                      for image, digit in zip(test_data[0], test_data[1]))#注意这里是测试集了
    print("Baseline classifier using average darkness of image.")
    print("%s of %s values correct." % (num_correct, len(test_data[1])))
#像素值表示图像的大小，像素坐标表示地址，灰度值表示地址中的值 白色灰度值=255  黑色灰度值=0
def avg_darknesses(training_data):
    """ Return a defaultdict whose keys are the digits 0 through 9.
    For each digit we compute a value which is the average darkness of
    training images containing that digit.  The darkness for any
    particular image is just the sum of the darknesses for each pixel."""
    digit_counts = defaultdict(int)#整型字典
    darknesses = defaultdict(float)#灰度值字典
    for image, digit in zip(training_data[0], training_data[1]):
        digit_counts[digit] += 1#记录每个数字出现的次数
        darknesses[digit] += sum(image)#记录每个数字的灰度值的和
    avgs = defaultdict(float)
    for digit, n in digit_counts.items():#返回的是（数字，频数）元组
        avgs[digit] = darknesses[digit] / n
    return avgs #返回每个数字的平均灰度值，是个字典形式

def guess_digit(image, avgs):
    """Return the digit whose average darkness in the training data is
    closest to the darkness of ``image``.  Note that ``avgs`` is
    assumed to be a defaultdict whose keys are 0...9, and whose values
    are the corresponding average darknesses across the training data."""
    darkness = sum(image)
    distances = {k: abs(v-darkness) for k, v in avgs.items()}
    return min(distances, key=distances.get)

if __name__ == "__main__":
    main()
