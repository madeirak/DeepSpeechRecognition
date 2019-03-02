import os
import tensorflow as tf
from utils import get_data, data_hparams


# 0.准备训练所需数据------------------------------
data_args = data_hparams()
data_args.shuffle = True
train_data = get_data(data_args)

# 1.声学模型训练-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams
am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)

if os.path.exists('logs_am/model.h5'):
    print('load acoustic model...')
    am.ctc_model.load_weights('logs_am/model.h5')

epochs = 10
batch_num = len(train_data.wav_lst) // train_data.batch_size

for k in range(epochs):
    print('this is the', k+1, 'th epochs trainning !!!')
    batch = train_data.get_am_batch()
    am.ctc_model.fit_generator(batch, steps_per_epoch=batch_num, epochs=1)#generator=batch，batch是一个生成器
                                                                          #steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch

am.ctc_model.save_weights('logs_am/model.h5')