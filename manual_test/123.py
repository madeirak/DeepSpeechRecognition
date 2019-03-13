import numpy as np
import scipy.io.wavfile as wav
from utils import decode_ctc
import tensorflow as tf
from utils import compute_fbank
from utils import get_data, data_hparams

data_args = data_hparams()
train_data = get_data(data_args)

'''将测试音频文件放于data/'''

# 1.声学模型-----------------------------------
from model_speech.cnn_ctc import Am, am_hparams

am_args = am_hparams()
am_args.vocab_size = len(train_data.am_vocab)
am = Am(am_args)
print('loading acoustic model...')
am.ctc_model.load_weights('G:/DeepSpeechRecognition/logs_am/model.h5')

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
with sess.as_default():
    latest = tf.train.latest_checkpoint('G:/DeepSpeechRecognition/logs_lm')
    saver.restore(sess, latest)


# 3.测试声音模型------------------------------------
filepath = 'data/A2_1.wav'
_, wavsignal = wav.read(filepath)

fbank = compute_fbank(filepath)
pad_fbank = np.zeros((fbank.shape[0]//8*8+8, fbank.shape[1]))

pad_fbank[:fbank.shape[0], :] = fbank
wav_data_lst = []
wav_data_lst.append(pad_fbank)

wav_lens = [len(data) for data in wav_data_lst]
wav_max_len = max(wav_lens)
new_wav_data_lst = np.zeros((len(wav_data_lst), wav_max_len, 200, 1))
wav_lens = np.array([leng//8 for leng in wav_lens])

new_wav_data_lst[0, :wav_data_lst[0].shape[0], :, 0] = wav_data_lst[0]


result = am.model.predict(new_wav_data_lst, steps=1)

_, text = decode_ctc(result, train_data.am_vocab)  # num2pny
text = ' '.join(text)
print('拼音结果：', text)


# 3.测试语言模型------------------------------------
with sess.as_default():
    line = text
    line = line.strip('\n').split(' ')
    x = np.array([train_data.pny_vocab.index(pny) for pny in line])#pny2id
    x = x.reshape(1, -1)
    preds = sess.run(lm.preds, {lm.x: x})
    got = ''.join(train_data.han_vocab[idx] for idx in preds[0])  #id2hanzi
    print('汉字结果：',got)
sess.close()