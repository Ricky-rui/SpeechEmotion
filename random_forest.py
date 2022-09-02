import pandas as pd
from decision_tree import DecisionTreeClassifier
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import numpy
from scipy.io import wavfile
from scipy.fftpack import dct
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def transfer_rate(orig_confusion_matrix):
    rate_matrix = []
    for line in orig_confusion_matrix:
        sum = 0
        line_matrix = []
        for value in line:
            sum = sum + value
        for value in line:
            worth = round(value / sum * 100, 2)
            line_matrix.append(str(worth) + "%")
        rate_matrix.append(line_matrix)
    return rate_matrix



def segment(vectors):
    sum_vectors = []
    for vector in vectors:
        s_vector = []
        for value in vector:
            value = int(value * 10)
            s_vector.append(value)
        sum_vectors.append(s_vector)
    return sum_vectors

def mfcc_trans_vectors(position):
    mfcc_s_rate, signal = wavfile.read(position)
    signal = signal[0:int(3.5 * mfcc_s_rate)]
    mfcc_frame_stride = 0.01
    mfcc_frame_size = 0.025
    mfcc_preemphasis = 0.97
    mfcc_emphasizedsignal = numpy.append(signal[0], signal[1:] - mfcc_preemphasis * signal[:-1])
    mfcc_frame_length, frame_step = mfcc_frame_size * mfcc_s_rate, mfcc_frame_stride * mfcc_s_rate  # Convert from seconds to samples
    mfcc_signal_length = len(mfcc_emphasizedsignal)
    mfcc_frame_length = int(round(mfcc_frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(
        numpy.ceil(
            float(numpy.abs(mfcc_signal_length - mfcc_frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    pad_mfcc_signal_length = num_frames * frame_step + mfcc_frame_length
    z = numpy.zeros((pad_mfcc_signal_length - mfcc_signal_length))
    pad_signal = numpy.append(mfcc_emphasizedsignal, z)
    # Populate the signal to ensure that all frames have the same number of samples without truncating any samples in the original signal

    indices = numpy.tile(numpy.arange(0, mfcc_frame_length), (num_frames, 1)) + numpy.tile(
        numpy.arange(0, num_frames * frame_step, frame_step), (mfcc_frame_length, 1)).T
    frames = pad_signal[indices.astype(numpy.int32, copy=False)]
    frames *= numpy.hamming(mfcc_frame_length)
    NFFT = 512
    mfcc_mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mfcc_mag_frames) ** 2))  # Power Spectrum
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * numpy.log10(1 + (mfcc_s_rate / 2) / 700))  # Convert Hz to Mel
    mfcc_mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10 ** (mfcc_mel_points / 2595) - 1))  # Convert Mel to Hz
    mfcc_bin = numpy.floor((NFFT + 1) * hz_points / mfcc_s_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(mfcc_bin[m - 1])  # left
        f_m = int(mfcc_bin[m])  # center
        f_m_plus = int(mfcc_bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - mfcc_bin[m - 1]) / (mfcc_bin[m] - mfcc_bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (mfcc_bin[m + 1] - k) / (mfcc_bin[m + 1] - mfcc_bin[m])
    mfcc_filter_banks = numpy.dot(pow_frames, fbank.T)
    mfcc_filter_banks = numpy.where(mfcc_filter_banks == 0, numpy.finfo(float).eps, mfcc_filter_banks)  # Numerical Stability
    mfcc_filter_banks = 20 * numpy.log10(mfcc_filter_banks)  # dB
    num_ceps = 12
    mfcc_vector = dct(mfcc_filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13


    mfcc_cep_lifter = 22
    (nframes, ncoeff) = mfcc_vector.shape
    n = numpy.arange(ncoeff)
    lift = 1 + (mfcc_cep_lifter / 2) * numpy.sin(numpy.pi * n / mfcc_cep_lifter)
    mfcc_vector *= lift
    mfcc_filter_banks -= (numpy.mean(mfcc_filter_banks, axis=0) + 1e-8)
    mfcc_vector -= (numpy.mean(mfcc_vector, axis=0) + 1e-8)
    return mfcc_vector

def read_IEMocap():
    pca = PCA(n_components=1)
    iemocap_data_DIR = r"IEMOCAP"

    mfcc_vectors = []
    labels = []
    import os
    for speaker in os.listdir(iemocap_data_DIR):

        if speaker[0] == 'S':
            emo_labledir = os.path.join(iemocap_data_DIR, speaker, "dialog/EmoEvaluation")
            wav_subdir = os.path.join(iemocap_data_DIR, speaker, "sentences/wav")
            for sess in os.listdir(wav_subdir):
                if "Ses" not in sess:
                    continue
                lable_text = emo_labledir+"/"+sess+".txt"
                mfcc_emotion_lable = {}

                with open(lable_text,'r') as txt_read:
                    while True:
                        line = txt_read.readline()
                        if not line:
                            break
                        if (line[0] == '['):
                            t = line.split()
                            mfcc_emotion_lable[t[3]] = t[4]

                for key in mfcc_emotion_lable.keys():
                    #print(key)
                    newkeys = key.split("_")[:-1]
                    file_directory = newkeys[0]
                    for i in range(1, len(newkeys)):
                        file_directory = file_directory + "_" + newkeys[i]
                    file_position = wav_subdir + "/" + file_directory + "/" + key + ".wav"
                    #signal, sample_rate = librosa.load(file_position, sr=8000)
                    #mfcc_v = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=12, dct_type=2, norm='ortho')
                    #mfcc提取
                    mfcc_v = mfcc_trans_vectors(file_position).T
                    reduced_x = pca.fit_transform(mfcc_v)
                    flat_x = reduced_x.reshape(12)
                    mfcc_vectors.append(flat_x)
                    labels.append(mfcc_emotion_lable[key])

        labeldict = {'neu': 0, 'ang': 1, 'sad': 2, 'hap': 3}


        newlabels = []
        new_vectors = []
        for i in range(0, len(labels)):
            if labeldict.__contains__(labels[i]):
                new_vectors.append(mfcc_vectors[i])
                newlabels.append(labeldict[labels[i]])

        #state = np.random.get_state()
        #np.random.shuffle(mfcc_vectors)
        #np.random.set_state(state)
        #np.random.shuffle(newlabels)
        #test_position = len(mfcc_vectors) - int(len(mfcc_vectors) / 5)
        #train_vectors = mfcc_vectors[:test_position]
        #train_labels = newlabels[:test_position]
        #test_vectors = mfcc_vectors[test_position:]
        #test_labels = newlabels[test_position:]
        x_train, x_test, y_train, y_test = train_test_split(new_vectors, newlabels, test_size=0.3, random_state=5)

    return x_train, x_test, y_train, y_test


class RandomForestClassifier(object):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None,
                 d=None,
                 random_state=0):

        self.estimators = [DecisionTreeClassifier(criterion, max_depth, d,
                                                  random_state + i) for i in
                           range(n_estimators)]
        self.random_state = random_state

    def fit(self, X_vectors, y_lables):
        if isinstance(X_vectors, np.ndarray):
            X_vectors = pd.DataFrame(X_vectors)

        rgen = np.random.RandomState(self.random_state)
        N, _ = X_vectors.shape
        indices = np.arange(N)
        if isinstance(y_lables, (pd.DataFrame, pd.Series)):
            y_lables = np.squeeze(y_lables.values)
        for estimator in self.estimators:
            sampled_indices = rgen.choice(indices, size=N, replace=True)
            vectors_X = X_vectors.iloc[sampled_indices]
            vectors_y = y_lables[sampled_indices]
            estimator.fit(vectors_X, vectors_y)

    def predict(self, X_vectors):
        '''
        Returns predicted categories of `X`
        Args:
            X: pandas.DataFrame of numpy.ndarray
                input data
        Returns:
            pred_y: predicted categories of `X`
        '''
        all_preds = []
        for estimator in self.estimators:
            all_preds.append(estimator.predict(X_vectors))
        all_preds = np.asarray(all_preds)
        all_preds = np.split(all_preds, all_preds.shape[1], axis=1)
        predict = []
        for p in all_preds:
            label, count = np.unique(p.squeeze(), return_counts=True)
            predict.append(label[np.argmax(count)])
        predict = np.asarray(predict)
        return predict


if __name__ == '__main__':

    train_vectors, test_vectors, train_labels, test_labels = read_IEMocap()
    mm = MinMaxScaler()
    train_vectors = mm.fit_transform(train_vectors)
    test_vectors = mm.fit_transform(test_vectors)
    train_vectors = segment(train_vectors)
    test_vectors = segment(test_vectors)

    model = RandomForestClassifier(n_estimators=10, max_depth=10)
    model.fit(np.array(train_vectors), np.array(train_labels))
    predict = model.predict(np.array(test_vectors))

    print(metrics.accuracy_score(test_labels, predict))
    print(metrics.precision_score(test_labels, predict, average='macro'))
    print(metrics.recall_score(test_labels, predict, average='macro'))
    print(metrics.f1_score(test_labels, predict, average='weighted'))
    print(transfer_rate(metrics.confusion_matrix(test_labels, predict)))


    print("---------------------- roc  ------------------------")
    from sklearn.preprocessing import label_binarize
    import matplotlib.pyplot as plt

    y = label_binarize(test_labels, classes=[0, 1, 2, 3])
    y_predict = label_binarize(predict, classes=[0, 1, 2, 3])
    #print(y)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    from sklearn.metrics import roc_curve, auc
    from scipy import interp

    for i in range(4):
        fpr[i], tpr[i], _ = roc_curve(y_predict[:, i], y[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    fpr["micro"], tpr["micro"], _ = roc_curve(y_predict.ravel(), y.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(4)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(4):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])


    mean_tpr /= 4

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])


    plt.figure()
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    from itertools import cycle

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(4), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()



