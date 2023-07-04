# import librosa
# import librosa.display
import matplotlib.pyplot as plt
from python_speech_features import logfbank
import scipy.io.wavfile as wav
# 加载音频文件
(sample_rate,waveform) = wav.read(r'D:\learn\learn\asr_socr\data_aishell\data_aishell\wav\train\S0002\BAC009S0002W0123.wav')
#waveform, sample_rate = librosa.load(r'D:\learn\learn\asr_socr\data_aishell\data_aishell\wav\train\S0002\BAC009S0002W0123.wav')
# 提取fbank特征
fbank_features = logfbank(waveform,sample_rate,nfilt=80)


# 将fbank特征转换为图片
plt.imshow(fbank_features.T, aspect='auto', origin='lower')
plt.colorbar(format='%+2.0f dB')
plt.xlabel('Frame')
plt.ylabel('Filter Banks')
plt.title('FBank Features')
plt.show()

# 绘制波形图
# plt.figure(figsize=(12, 4))
# librosa.display.waveshow(waveform, sr=sample_rate)
# #librosa.display.waveplot(waveform, sr=sample_rate)
# plt.title('Waveform')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()