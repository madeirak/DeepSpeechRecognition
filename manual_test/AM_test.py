import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from utils import decode_ctc,compute_fbank
import os

# 0.准备解码所需字典
from utils import get_data, data_hparams
data_args = data_hparams()
data_args.data_type = 'train'
data_args.self_wav = True
data_args.thchs30 = False
data_args.aishell = False
data_args.prime = False
data_args.stcmd = False
train_data = get_data(data_args)

# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

#不打印警告
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('G:/DeepSpeechRecognition/logs_am/model_self.h5')#从绝对路径的检查点恢复权重数据


import matplotlib.pyplot as plt
filepath = 'data/5_.wav'

_, wavsignal = wav.read(filepath)
#plt.plot(wavsignal)
#plt.show()

fbank = compute_fbank(filepath)
#plt.imshow(fbank.T, origin = 'lower')
#plt.show()

pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))  #“//”整除，向下取整，“//”与“*”优先级相同，从左往右计算
																				#结果是a.shape[0]即每个元素的帧长可以被8整除

pad_fbank[:fbank.shape[0], :] = fbank
wav_data_lst = []
wav_data_lst.append(pad_fbank)

wav_lens = [len(data) for data in wav_data_lst]
wav_max_len = max(wav_lens)
new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
wav_lens = np.array([leng//8 for leng in wav_lens])

new_wav_data_lst[0, :wav_data_lst[0].shape[0], :, 0] = wav_data_lst[0]

#new_wav_data_lst = tf.expand_dims(new_wav_data_lst, 0)#3d->4d

result = am.model.predict(new_wav_data_lst, steps=1)#steps预测周期结束前的总步骤数(样品批次)，predict返回numpy数组类型的预测
_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
#_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
text = ' '.join(text)  # 以空格为分隔符合将多元素列表text合并成一个字符串
print('文本结果：', text)
