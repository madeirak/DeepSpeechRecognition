import tensorflow as tf
import numpy as np
import os

# 0.准备解码所需字典，参数需和训练一致，也可以将字典保存到本地，直接进行读取
from utils import get_data, data_hparams
data_args = data_hparams()
train_data = get_data(data_args)


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
    latest = tf.train.latest_checkpoint('G:/DeepSpeechRecognition/logs_lm')#查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
    saver.restore(sess, latest)#restore(sess,save_path)，需要启动图表的会话
                               # 该save_path参数通常是先前从save()调用或调用返回的值latest_checkpoint()


#限定最大GPU用量
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5# 占用GPU90%的显存
config.gpu_options.allow_growth = True
session = tf.Session(config = config)

#不打印警告
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# 3.手动测试语言模型------------------------------------
with sess.as_default():
    for i in range(10):
        line = input('输入测试拼音: ')
        if line == 'exit': break
        line = line.strip('\n').split(' ')
        x = np.array([train_data.pny_vocab.index(pny) for pny in line])#pny2id
        x = x.reshape(1, -1)
        preds = sess.run(lm.preds, {lm.x: x})
        got = ''.join(train_data.han_vocab[idx] for idx in preds[0])  #id2hanzi
        print(got)
sess.close()