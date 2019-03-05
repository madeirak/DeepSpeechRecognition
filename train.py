import os
import tensorflow as tf
from utils import get_data, data_hparams
from keras.callbacks import ModelCheckpoint


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'train'
data_args.data_path = '../dataset/'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = True
data_args.stcmd = True
data_args.batch_size = 4
data_args.data_length = 10
# data_args.data_length = None
data_args.shuffle = True
train_data = get_data(data_args)

# 0.准备验证所需数据------------------------------
data_args = data_hparams()
data_args.data_type = 'dev'
data_args.data_path = '../dataset/'
data_args.thchs30 = True
data_args.aishell = True
data_args.prime = False
data_args.stcmd = False
data_args.batch_size = 4
# data_args.data_length = None
data_args.data_length = 10
data_args.shuffle = True
dev_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am_args.gpu_nums = 1
am_args.lr = 0.0008
am_args.is_training = True
am = Am(am_args)

if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

epochs = 10
batch_num = len(train_data.wav_lst) // train_data.batch_size


# checkpoint
ckpt = "model_{epoch:02d}-{val_acc:.2f}.hdf5"#字符串中包含格式化选项（named formatting options），epoch_num和validation_accuracy验证准确率
                                             #“:02d”表示右对齐长度为2
checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_acc',
                             save_weights_only=False, verbose=1, save_best_only=True)
                            #若出现”./”开头的参数，会从”./”开头的参数的上一个参数开始拼接。
                            #monitor='val_acc'监测验证准确率
                            #verbose详细模式，0为不打印输出信息，1位打印输出
                            #save_weights_only=False，不只保存权重而是整个模型

#
# for k in range(epochs):
#     print('this is the', k+1, 'th epochs trainning !!!')
#     batch = train_data.get_am_batch()
#     dev_batch = dev_data.get_am_batch()
#     am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint],
#     workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)

batch = train_data.get_am_batch()
dev_batch = dev_data.get_am_batch()

am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=10, callbacks=[checkpoint],
                           workers=1, use_multiprocessing=False, validation_data=dev_batch, validation_steps=200)
 #（训练数据生成器实例，一个epoch有几个batch(batch_num)，epoch，回调函数，最大进程数，不用多线程，验证数据生成器，设置验证多少次数据后取平均值作为此epoch训练后的效果）

am.ctc_model.save_weights('logs_am/model.h5')#保存所有层的权重，“h5”表示HDF5格式保存
                                            #默认情况下，会以 TensorFlow 检查点文件格式保存模型的权重。权重也可以另存为 Keras HDF5 格式


# 2.语言模型训练-------------------------------------------
from model_language.transformer import Lm, lm_hparams
lm_args = lm_hparams()
lm_args.num_heads = 8
lm_args.num_blocks = 6
lm_args.input_vocab_size = len(train_data.pny_vocab)
lm_args.label_vocab_size = len(train_data.han_vocab)
lm_args.max_length = 100
lm_args.hidden_units = 512
lm_args.dropout_rate = 0.2
lm_args.lr = 0.0003
lm_args.is_training = True
lm = Lm(lm_args)

epochs = 10
with lm.graph.as_default():
    saver =tf.train.Saver()
with tf.Session(graph=lm.graph) as sess:#创建会话对象sess。会话会封装TensorFlow运行时的状态，并运行TensorFlow操作。
    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())#初始化图中的所有变量（可训练参数）
    add_num = 0
    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')
        add_num = int(latest.split('_')[-1])#分隔后保存为列表，取最后一个
        saver.restore(sess, latest)
    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())

    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)#run方法的feed_dict参数为占位符提供具体的值
                                                                         #因为run的返回和输入有相同的布局，又feed_dict是一个指令
                                                                         #而不是一个张量，不会返回一个值，所以用“_”
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)
                writer.add_summary(rs, k * batch_num + i)
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)
    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))
    writer.close()