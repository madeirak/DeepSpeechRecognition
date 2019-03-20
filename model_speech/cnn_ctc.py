import keras
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D
from keras.layers import Reshape, Dense, Dropout, Lambda
from keras.optimizers import Adam
from keras import backend as K
from keras.models import Model
from keras.utils import multi_gpu_model
import tensorflow as tf


def am_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        MS_OUTPUT_SIZE=127,
        lr = 0.0008,
        gpu_nums = 1,
        is_training = True)
    return params


# =============================搭建模型====================================

class Am():#通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
    """docstring for Amodel."""
    def __init__(self, args):#将类实例的属性应用到后续的层创建中
        self.MS_OUTPUT_SIZE = args.MS_OUTPUT_SIZE
        self.gpu_nums = args.gpu_nums
        self.lr = args.lr
        self.is_training = args.is_training
        self._model_init()
        if self.is_training:
            self._ctc_init()
            self.opt_init()

    def _model_init(self):#创建层，并定义前向传播
        self.inputs = Input(name='the_inputs', shape=(None, 200, 1))#shape限定三维张量，第一维是batch_num？
                                                                    #Input用于实例化keras张量，产生象征性的张量，类似于占位符，name是它的名字

        self.h1 = cnn_cell(32, self.inputs)  #第一个参数size，输出空间的维数，即卷积层中的滤波器数目
        self.h2 = cnn_cell(64, self.h1)
        self.h3 = cnn_cell(128, self.h2)
        self.h4 = cnn_cell(128, self.h3, pool=False)
        self.h5 = cnn_cell(128, self.h4, pool=False)
        # 200 / 8 * 128 = 3200 权重数
        self.h6 = Reshape((-1, 3200))(self.h5)#-1是占位符，这一维的长度根据其他维而定
        self.h6 = Dropout(0.2)(self.h6)
        self.h7 = dense(256)(self.h6)
        self.h7 = Dropout(0.2)(self.h7)
        self.outputs = dense(self.MS_OUTPUT_SIZE, activation='softmax')(self.h7)
        self.model = Model(inputs=self.inputs, outputs=self.outputs)#实例化上述自定义模型
        self.model.summary()

    def _ctc_init(self):#计算ctc损失的模型自定义
        self.labels = Input(name='the_labels', shape=[None], dtype='float32')
        self.input_length = Input(name='input_length', shape=[1], dtype='int64')
        self.label_length = Input(name='label_length', shape=[1], dtype='int64')

        self.loss_out = Lambda(ctc_lambda, output_shape=(1,), name='ctc')\
        ([self.labels, self.outputs, self.input_length, self.label_length])#“Lambda”API用于将任意表达式包装为“层”对象
                                                                           #反斜杠“\”是用来连接多行较长的语句
        self.ctc_model = Model(inputs=[self.labels, self.inputs,
            self.input_length, self.label_length], outputs=self.loss_out)#Model从输入（inputs）和输出（outputs）创建模型

    def opt_init(self):#优化器初始化
        opt = Adam(lr = self.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01, epsilon = 10e-8)#eplison模糊因子，decay每次更新时lr的下降

        if self.gpu_nums > 1:#多GPU
            self.ctc_model=multi_gpu_model(self.ctc_model,gpus=self.gpu_nums)

        self.ctc_model.compile(loss={'ctc': lambda y_true, output: output}, optimizer=opt,metrics = ['accuracy'])#单GPU
                                                                                    #complie函数编译模型model以供训练，编译时指明loss和优化器
                                                                                    #y_true为真实数据标签（对应于上面的y_pred输出的预测值）                                                                                  #






# ============================模型组件=================================
def ctc_lambda(args):
    labels, y_pred, input_length, label_length = args   #labels包含标签的张量；y_pred包含softmax输出的张量
                                                        #input_length是y_pred中每个批元素的序列长度
    y_pred = y_pred[:, :, :]#截取y_pred前三维
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)#在每个批上运行ctc损失算法
                                                                       #ctc_batch_cost()返回具有(item_num_of_batch，1)形状的张量,
                                                                       #包含每个item的CTC损失

def conv2d(size):
    return Conv2D(size, (3,3), use_bias=True, activation='relu',
        padding='same', kernel_initializer='he_normal')


def norm(x):
    return BatchNormalization(axis=-1)(x)#设定标准化层并传入x


def maxpool(x):
    return MaxPooling2D(pool_size=(2,2), strides=None, padding="valid")(x)#valid表示不padding，步长strides=None表示不重叠且紧贴，所以维度会减半


def dense(units, activation="relu"):
    return Dense(units, activation=activation, use_bias=True,
        kernel_initializer='he_normal')#全连接层units输出维度


# x.shape=(none, none, none)
# output.shape = (1/2, 1/2, 1/2)
def cnn_cell(size, x, pool=True):# #第一个参数size，输出空间的维数，即卷积层中的滤波器数目
    x = norm(conv2d(size)(x))
    x = norm(conv2d(size)(x))
    if pool:
        x = maxpool(x)           #此卷积核的两个卷积层不减少维度，最大池化使各维度减半！
    return x



