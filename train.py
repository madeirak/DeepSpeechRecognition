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
#data_args.data_length = None #作者做实验时写小一些看效果用的，正常使用设为None，用于截取部分的lst数据
#data_args.shuffle = True
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
data_args.data_length = None #作者做实验时写小一些看效果用的，正常使用设为None
#data_args.data_length = 10
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
#checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_acc',
#                             save_weights_only=False, verbose=1, save_best_only=True)
                            #若出现”./”开头的参数，会从”./”开头的参数的上一个参数开始拼接。
                            #monitor='val_acc'监测验证准确率
                            #verbose详细模式，0为不打印输出信息，1位打印输出
                            #save_weights_only=False，不只保存权重而是整个模型
checkpoint = ModelCheckpoint(os.path.join('./checkpoint', ckpt), monitor='val_loss',
                             save_weights_only=False, verbose=1, save_best_only=True)


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
with lm.graph.as_default():#as_default()，将此图作为运行环境的默认图
    saver =tf.train.Saver()#tf.train.Saver 构造函数会针对图中所有变量将 save和restore操作 添加到图中。
with tf.Session(graph=lm.graph) as sess:#为指定图创建会话对象sess。会话会封装TensorFlow运行时的状态，并运行TensorFlow操作。
                                        #由于 tf.Session 拥有物理资源（例如 GPU 和网络连接），因此通常（在with代码块中）
                                        #用作上下文管理器，并在您退出代码块时自动关闭会话。

    merged = tf.summary.merge_all()#将之前创建的所有总结节点（tf.summary.scalar），合并为一个操作，方便之后运行生成汇总数据
    sess.run(tf.global_variables_initializer())#初始化图中的所有变量（可训练参数）
    add_num = 0
    if os.path.exists('logs_lm/checkpoint'):
        print('loading language model...')
        latest = tf.train.latest_checkpoint('logs_lm')#查找最新保存的检查点文件的文件名，latest_checkpoint(checkpoint_dir)
        add_num = int(latest.split('_')[-1])#分隔后保存为列表，取最后一个
        saver.restore(sess, latest)#restore(sess,save_path)，需要启动图表的会话。要恢复的变量不必初始化，因为恢复本身就是一种初始化变量的方法。
                                   #该save_path参数通常是先前从save()调用或调用返回的值 latest_checkpoint()

    writer = tf.summary.FileWriter('logs_lm/tensorboard', tf.get_default_graph())#FileWriter（logdir,graph）
                                                                                 #所有事件都会写到logdir所指的目录下
                                                                                 #接收到Graph对象，则会将图与张量形状信息一起可视化

    for k in range(epochs):
        total_loss = 0
        batch = train_data.get_lm_batch()
        for i in range(batch_num):
            input_batch, label_batch = next(batch)
            feed = {lm.x: input_batch, lm.y: label_batch}
            cost,_ = sess.run([lm.mean_loss,lm.train_op], feed_dict=feed)#run方法的feed_dict参数为占位符提供具体的值
                                                                         #因为run的返回和输入有相同的布局，又feed_dict是一个指令
                                                                         #而不是一个张量，不会返回一个值，所以用“_”
                                                                         #op:optimizer
                                                                         #mean_loss:batch_mean_loss
            total_loss += cost
            if (k * batch_num + i) % 10 == 0:
                rs=sess.run(merged, feed_dict=feed)#运行merged操作，收集汇总数据
                writer.add_summary(rs, k * batch_num + i)#训练时添加总结
        print('epochs', k+1, ': average loss = ', total_loss/batch_num)

    saver.save(sess, 'logs_lm/model_%d' % (epochs + add_num))#将图中训练后的变量保存到检查点文件中
    writer.close()


