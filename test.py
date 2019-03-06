#coding=utf-8
import os
import difflib
import tensorflow as tf
import numpy as np
from utils import decode_ctc, GetEditDistance


# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
train_data = get_data(data_args)


# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('logs_am/model.h5')#从检查点恢复权重数据

# 2.语言模型-------------------------------------------
from model_language.transformer import Lm, lm_hparams

lm_args = lm_hparams()
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.dropout_rate = 0.
print('loading language model...')
lm = Lm(lm_args)
sess = tf.Session(graph=lm.graph)
with lm.graph.as_default():
    saver =tf.train.Saver()
with sess.as_default():#创建默认会话
    latest = tf.train.latest_checkpoint('logs_lm')#查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest)#restore(sess,save_path)，需要启动图表的会话。
                               # 该save_path参数通常是先前从save()调用或调用返回的值latest_checkpoint()

# 3. 准备测试所需数据， 不必和训练数据一致，通过设置data_args.data_type测试，
#    此处应设为'test'，我用了'train'因为演示模型较小，如果使用'test'看不出效果，
#    且会出现未出现的词。？？？
data_args.data_type = 'train'
data_args.shuffle = False
data_args.batch_size = 1
test_data = get_data(data_args)

# 4. 进行测试-------------------------------------------
am_batch = test_data.get_am_batch()
word_num = 0
word_error_num = 0
for i in range(10):
    print('\n the ', i, 'th example.')
    # 载入训练好的模型，并进行识别
    inputs, _ = next(am_batch)#yield inputs, outputs
    x = inputs['the_inputs']#'the_inputs': pad_wav_data
    y = test_data.pny_lst[i]
    result = am.model.predict(x, steps=1)#steps预测周期结束前的总步骤数(样品批次)，predict返回numpy数组类型的预测

    # 将数字结果转化为文本结果
    _, text = decode_ctc(result, train_data.am_vocab)#num2word
    text = ' '.join(text)#以空格为分隔符合将多元素列表text合并成一个字符串
    print('文本结果：', text)
    print('原文结果：', ' '.join(y))#以空格为分隔符将多元素列表y合并成一个字符串

    with sess.as_default():
        text = text.strip('\n').split(' ')#strip返回移除字符串头尾指定的字符生成的新字符串；split返回指定分隔符分隔后的字符串列表
        x = np.array([train_data.pny_vocab.index(pny) for pny in text])#pny2num
        x = x.reshape(1, -1)#排成一行
        preds = sess.run(lm.preds, {lm.x: x})
        label = test_data.han_lst[i]
        got = ''.join(train_data.han_vocab[idx] for idx in preds[0])#num2hanzi
        print('原文汉字：', label)
        print('识别结果：', got)
        word_error_num += min(len(label), GetEditDistance(label, got))
        word_num += len(label)
print('词错误率：', word_error_num / word_num)
sess.close()
