import numpy as np
import scipy.io.wavfile as wav
from scipy.fftpack import fft
from utils import decode_ctc
import os

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

#不打印警告
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

am_args = am_hparams()
am_args.vocab_size = len(dev_data.am_vocab)
#am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('G:/DeepSpeechRecognition/logs_am/model.h5')#从绝对路径的检查点恢复权重数据

# 获取信号的时频图
def compute_fbank(file):
	x=np.linspace(0, 400 - 1, 400, dtype = np.int64)
	w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1) ) # 汉明窗
	fs, wavsignal = wav.read(file)
	# wav波形 加时间窗以及时移10ms
	time_window = 25 # 单位ms
	window_length = fs / 1000 * time_window # 计算窗长度的公式，目前全部为400固定值
	wav_arr = np.array(wavsignal)
	wav_length = len(wavsignal)
	range0_end = int(len(wavsignal)/fs*1000 - time_window) // 10 # 计算循环终止的位置，也就是最终生成的窗数
	data_input = np.zeros((range0_end, 200), dtype = np.float) # 用于存放最终的频率特征数据
	data_line = np.zeros((1, 400), dtype = np.float)
	for i in range(0, range0_end):
		p_start = i * 160
		p_end = p_start + 400
		data_line = wav_arr[p_start:p_end]
		data_line = data_line * w # 加窗
		data_line = np.abs(fft(data_line))
		data_input[i]=data_line[0:200] # 设置为400除以2的值（即200）是取一半数据，因为是对称的
	data_input = np.log(data_input + 1)
	#data_input = data_input[::]
	return data_input


import matplotlib.pyplot as plt
filepath = 'data/test.wav'

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
_, text = decode_ctc(result, dev_data.am_vocab)  # num2pny
#_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
text = ' '.join(text)  # 以空格为分隔符合将多元素列表text合并成一个字符串
print('文本结果：', text)