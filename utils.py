import os
import difflib
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
from tqdm import tqdm
from scipy.fftpack import fft
#from python_speech_features import mfcc
from random import shuffle
from keras import backend as K

def data_hparams():
    params = tf.contrib.training.HParams(
        # vocab
        data_type='train',
        data_path='G:/yinpin_data/',
        self_wav=True,
        thchs30=True,
        aishell=True,
        prime=True,
        stcmd=True,
        batch_size=1,
        data_length=10,
        shuffle=True)
    return params


class get_data():
    def __init__(self, args):
        self.data_type = args.data_type
        self.data_path = args.data_path
        self.self_wav = args.self_wav
        self.thchs30 = args.thchs30
        self.aishell = args.aishell
        self.prime = args.prime
        self.stcmd = args.stcmd
        self.data_length = args.data_length
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.source_init()

    def source_init(self):
        print('get source list...')
        read_files = []
        if self.data_type == 'train':
            if self.self_wav == True:
                read_files.append('self_wav_train.txt')
            if self.thchs30 == True:
                read_files.append('thchs_train.txt')
            if self.aishell == True:
                read_files.append('aishell_train.txt')
            if self.prime == True:
                read_files.append('prime.txt')
            if self.stcmd == True:
                read_files.append('stcmd.txt')
        elif self.data_type == 'dev':                           #development set    验证集dev
            if self.thchs30 == True:
                read_files.append('thchs_dev.txt')
            if self.self_wav == True:
                read_files.append('self_wav_dev.txt')
            if self.aishell == True:
                read_files.append('aishell_dev.txt')
        elif self.data_type == 'test':
            if self.thchs30 == True:
                read_files.append('thchs_test.txt')
            if self.aishell == True:
                read_files.append('aishell_test.txt')
        self.wav_lst = []
        self.pny_lst = []
        self.han_lst = []
        for file in read_files:
            print('load ', file, ' data...')
            sub_file = 'data/' + file
            with open(sub_file, 'r', encoding='UTF-8-sig') as f:
                data = f.readlines()
            for line in tqdm(data):
                wav_file, pny, han = line.split('\t')
                self.wav_lst.append(wav_file)
                self.pny_lst.append(pny.split(' '))
                self.han_lst.append(han.strip('\n'))  #一行字作为字符串作为表的一个元素
        if self.data_length:
            self.wav_lst = self.wav_lst[:self.data_length]  #如需，截取设定长度数据的lst中数据
            self.pny_lst = self.pny_lst[:self.data_length]
            self.han_lst = self.han_lst[:self.data_length]
        print('make am vocab...')
        self.am_vocab = self.mk_am_vocab(self.pny_lst)
        print('make lm pinyin vocab...')
        self.pny_vocab = self.mk_lm_pny_vocab(self.pny_lst)
        print('make lm hanzi vocab...')
        self.han_vocab = self.mk_lm_han_vocab(self.han_lst)

    def get_am_batch(self):
        shuffle_list = [i for i in range(len(self.wav_lst))]
        while 1:
            if self.shuffle == True:
                shuffle(shuffle_list)
            for i in range(len(self.wav_lst) // self.batch_size):
                wav_data_lst = []
                label_data_lst = []
                begin = i * self.batch_size
                end = begin + self.batch_size
                sub_list = shuffle_list[begin:end]
                for index in sub_list:  #遍历batch中每句
                    fbank = compute_fbank(self.data_path + self.wav_lst[index])
                    pad_fbank = np.zeros((fbank.shape[0] // 8 * 8 + 8, fbank.shape[1]))  #fbank.shape[1] = 200
                                                                                         #结果是fbank.shape[0]即每个元素的帧长可以被8整除
                    pad_fbank[:fbank.shape[0], :] = fbank
                    label = self.pny2id(self.pny_lst[index], self.am_vocab)

                    label_ctc_len = self.ctc_len(label) #label:一句的label  ctc_len:ctc输出长度
                    if pad_fbank.shape[0] // 8 >= label_ctc_len: #token_num > ctc_len  ctc_len？？？
                        wav_data_lst.append(pad_fbank)  #(batch_num,wav_len,200)
                        label_data_lst.append(label)
                pad_wav_data, input_length = self.wav_padding(wav_data_lst)  #pad_wav_data.shape = (len(wav_data_lst), wav_max_len, 200, 1)
                pad_label_data, label_length = self.label_padding(label_data_lst) #label_padding.shape = (len(label_data_lst), max_label_len)
                inputs = {'the_inputs': pad_wav_data,
                          'the_labels': pad_label_data,
                          'input_length': input_length,     #len(wav_data_lst) per batch
                          'label_length': label_length,     #len(label_data_lst) per batch
                          }
                outputs = {'ctc': np.zeros(pad_wav_data.shape[0], )}
                yield inputs, outputs

    def get_lm_batch(self):
        batch_num = len(self.pny_lst) // self.batch_size
        for k in range(batch_num):
            begin = k * self.batch_size
            end = begin + self.batch_size
            input_batch = self.pny_lst[begin:end]
            label_batch = self.han_lst[begin:end]
            max_len = max([len(line) for line in input_batch])
            #padding
            input_batch = np.array(
                [self.pny2id(line, self.pny_vocab) + [0] * (max_len - len(line)) for line in input_batch])
            label_batch = np.array(
                [self.han2id(line, self.han_vocab) + [0] * (max_len - len(line)) for line in label_batch])
            yield input_batch, label_batch

    def pny2id(self, line, vocab):
        return [vocab.index(pny) for pny in line]

    def han2id(self, line, vocab):
        return [vocab.index(han) for han in line]

    def wav_padding(self, wav_data_lst):
        wav_lens = [len(data) for data in wav_data_lst]
        wav_max_len = max(wav_lens)
        wav_lens = np.array([leng // 8 for leng in wav_lens])
        new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
        for i in range(len(wav_data_lst)):
            new_wav_data_lst[i, :wav_data_lst[i].shape[0], :, 0] = wav_data_lst[i]
        return new_wav_data_lst, wav_lens

    def label_padding(self, label_data_lst):
        label_lens = np.array([len(label) for label in label_data_lst])
        max_label_len = max(label_lens)
        new_label_data_lst = np.zeros((len(label_data_lst), max_label_len))
        for i in range(len(label_data_lst)):
            new_label_data_lst[i][:len(label_data_lst[i])] = label_data_lst[i]
        return new_label_data_lst, label_lens

    def mk_am_vocab(self, data):
        vocab = []
        for line in tqdm(data):
            line = line
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        vocab.append('_')
        return vocab

    def mk_lm_pny_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            for pny in line:
                if pny not in vocab:
                    vocab.append(pny)
        return vocab

    def mk_lm_han_vocab(self, data):
        vocab = ['<PAD>']
        for line in tqdm(data):
            line = ''.join(line.split(' '))
            for han in line:
                if han not in vocab:
                    vocab.append(han)
        return vocab

    def ctc_len(self, label):
        add_len = 0
        label_len = len(label)
        for i in range(label_len - 1):
            if label[i] == label[i + 1]:
                add_len += 1
        return label_len + add_len

'''
# 对音频文件提取mfcc特征
def compute_mfcc(file):
    fs, audio = wav.read(file)
    mfcc_feat = mfcc(audio, samplerate=fs, numcep=26)   #numcep梅尔倒谱系数个数
    mfcc_feat = mfcc_feat[::3]
    mfcc_feat = np.transpose(mfcc_feat)
    return mfcc_feat
'''

# 获取信号的时频图(语谱图，语音频谱图) 横轴时间，纵轴频率，第三维用颜色表示幅值
def compute_fbank(file):
    x = np.linspace(0, 400 - 1, 400, dtype=np.int64)
    w = 0.54 - 0.46 * np.cos(2 * np.pi * (x) / (400 - 1))  # 汉明窗
    fs, wavsignal = wav.read(file)  #wavsignal 是声音波形图按照某一刻度刻画出来的离散点幅度的值 [a,b,c,...]
    # wav波形 加时间窗以及时移10ms
    time_window = 25  # 单位ms
    wav_arr = np.array(wavsignal)
                                    #len(wavsignal) / fs * 1000 是wav的总毫秒数
    range0_end = int(len(wavsignal) / fs * 1000 - time_window) // 10 + 1 # 计算循环终止的位置，也就是最终生成的窗数
    data_input = np.zeros((range0_end, 200), dtype=np.float)  # 二维数组,一列一帧，帧长400，加窗后近似对称，取一半即200
    data_line = np.zeros((1, 400), dtype=np.float)
    for i in range(0, range0_end):
        p_start = i * 160  #一个帧移的采样点 16kHz x 10ms
        p_end = p_start + 400
        data_line = wav_arr[p_start:p_end]
        data_line = data_line * w  # 加窗  w.shape=(400,)
        data_line = np.abs(fft(data_line)) #fft结果为复数，加绝对之后只取实部
        data_input[i] = data_line[0:200]  # 取一半数据，因为是对称的
    data_input = np.log(data_input + 1)
    # data_input = data_input[::]
    return data_input  #shaoe = (帧数,200)


# word error rate------------------------------------
def GetEditDistance(str1, str2):        #编辑距离是指两个字串之间，由一个转成另一个所需的最少编辑操作的度量
                                        ##许可的编辑操作包括单个字符的替换，插入，删除
    leven_cost = 0
    s = difflib.SequenceMatcher(None, str1, str2)#	None处为丢弃函数，此处设置不丢弃。SequenceMatcher是构造函数，主要创建任何类型序列的比较对象

    '''
    get_opcodes函数每执行一次返回5个元素的元组，元组描述了从a序列变成b序列所经历的步骤。5个元素的元组表示为(tag, i1, i2, j1, j2)，其中tag表示动作，
    i1,i2分别表示序列a的开始和结束位置，j1，j2表示序列b的开始和结束位置。
    '''

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == 'replace':        #a[i1:i2] should be replaced by b[j1:j2]
            leven_cost += max(i2-i1, j2-j1)
        elif tag == 'insert':        #b[j1:j2] should be inserted at a[i1:i1].Note that i1==i2 in this case.
            leven_cost += (j2-j1)
        elif tag == 'delete':        #a[i1:i2] should be deleted. Note that j1==j2 in this case.
            leven_cost += (i2-i1)
    return leven_cost

# 定义解码器------------------------------------
def decode_ctc(num_result, num2word):
    result = num_result[:, :, :]
    in_len = np.zeros((1), dtype = np.int32)
    in_len[0] = result.shape[1]


    '''ctc_decode()返回一个二元组，第一项是一个一元素列表,其中包含了解码序列;第二项是(top_paths, )的张量，包含每个解码序列的log概率'''
    # ([<tf.Tensor 'SparseToDense:0' shape=(1, 13) dtype=int64>], <tf.Tensor 'CTCGreedyDecoder:3' shape=(1, 1) dtype=float32>)
    r = K.ctc_decode(result, in_len, greedy = True, beam_width=10, top_paths=1)# #集束搜索，result是y_pred，top_paths=1表示最终返回一条最可能的路径
    r1 = K.get_value(r[0][0])       #get_value输入变量，返回一个数组    取二元组r的第一项中的第一个
    #print('r1:', r1)
    r1 = r1[0]      # （1，解码序列长） ->  （解码序列长，）
    #print('r1[0]:',r1)
    text = []
    for i in r1:
        text.append(num2word[i])
    return r1, text


if (__name__ == '__main__') :
    pass