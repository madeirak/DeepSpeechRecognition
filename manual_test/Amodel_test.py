import numpy as np
import scipy.io.wavfile as wav
from utils import decode_ctc
import os
from utils import compute_fbank

#不打印警告
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


# 0.准备解码所需字典
from utils import get_data, data_hparams

#data_args = data_hparams()

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


#train_data = get_data(data_args)
# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(dev_data.am_vocab)
#am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('G:/DeepSpeechRecognition/logs_am/model.h5')#从绝对路径的检查点恢复权重数据

import matplotlib.pyplot as plt
filepath = 'data/A2_1.wav'

_, wavsignal = wav.read(filepath)#return fs,wavsignal
#plt.plot(wavsignal)
#plt.show()

fbank = compute_fbank(filepath)#fbank.shape = (帧数，200)
#plt.imshow(fbank.T, origin = 'lower')
#plt.show()

pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))  #结果是a.shape[0]即每个元素的帧长可以被8整除,np.zeros返回数组


pad_fbank[:fbank.shape[0], :] = fbank  #"data/A2_1.wav"shape=(1024,200)

wav_data_lst = []
wav_data_lst.append(pad_fbank)

wav_lens = [len(data) for data in wav_data_lst]
wav_max_len = max(wav_lens)#list2int      #"data/A2_1.wav"wav_max_len=1024

new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))


new_wav_data_lst[0, :wav_data_lst[0].shape[0], :, 0] = wav_data_lst[0]    #"data/A2_1.wav"shape=(1,1024,200，1）

#print('\nnew_wav_data_lst',new_wav_data_lst.shape,'\n',new_wav_data_lst)
#new_wav_data_lst = tf.expand_dims(new_wav_data_lst, 0)#3d->4d

result = am.model.predict(new_wav_data_lst, steps=1)#steps预测周期结束前的总步骤数(样品批次)，predict返回numpy数组类型的预测
_, text = decode_ctc(result, dev_data.am_vocab)  # num2pny
#_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
text = ' '.join(text)  # 以空格为分隔符合将多元素列表text合并成一个字符串
print('文本结果：', text)
