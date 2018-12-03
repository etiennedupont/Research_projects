# Import libraries

import pandas as pd
import librosa
import numpy as np
import os
import os.path as osp

from audio_pickler import load_pickled_audio


# Define parsers

## MFCC features
def parser_mfcc_raw(audio_file, sample_rate, n_mfcc):
    # Here we extract mfcc feature from data
    # n_mfcc is the length of the output array : 
    # we choose to breakdown the spectrum into n_mfcc variables
    mfccs = np.mean(librosa.feature.mfcc(y=audio_file,sr=sample_rate,n_mfcc=n_mfcc).T,axis=0)
    return pd.Series([mfccs])

def parser_mfcc(row, data_directory_path, n_mfcc=40, pickled_audio=False, trim=False, db_cutoff=3):
    file_name = osp.join(data_directory_path,row['fname'])
    try:
        # here kaiser_fast is a technique used for faster extraction
        if not pickled_audio:
            X, sample_rate = librosa.load(file_name,res_type='kaiser_fast')
        else:
            X, sample_rate = load_pickled_audio(row['fname'], data_directory_path)
        # we extract mfcc feature from data
        if trim:
            X = librosa.effects.trim(X, top_db=db_cutoff)[0]
            
        mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate,n_mfcc=n_mfcc).T,axis=0)
    except Exception as e:
        print('Error encountered while parsing the file:',file_name)  
        return 'None'
    feature = mfccs  
    return pd.Series([feature])


# Low-lvl features
def parser_rolloff_raw(y, sr, percentiles, n_frames=None, hop_size=15e-3, window_size=25e-3, verbose=False):
    """
    An extractor for the roll-off frequencies. The user can either specify a number of frames to divide the signal in,
    OR manually set the hop size and the window size (but not both). By default, the hop_size is set.
    :param y: the audio file extracted
    :param sr: the sample rate
    :param percentiles: the percentiles to compute. Can be either a float or a list
    (e.g. percentiles = [0.25, 0.5, 0.75] to compute the median and quartiles)
    :param n_frames: the number of frames the audio file is expected to be divided in.
    :param hop_size: the hop size in seconds
    :param window_size: the window size in seconds
    :param verbose: the verbosity level of the function
    :return: a matrix len(percentiles) * n_frames, in which the ith column stores the energy percentiles for
    the ith frame. if percentiles is a float, return an array in which the ith element stores the energy percentile for
    the ith frame.
    """
    if n_frames is not None:  # If n_frames is specified
        assert n_frames >= 1, 'The \'n_frames\' argument should be a strictly positive integer.'
        if verbose:
            print('n_frames specified. Using a fixed number of frames n_frames={}'.format(n_frames))
        if n_frames > 1:
            hop_len = len(y) // (n_frames - 1)
        elif n_frames == 1:
            hop_len = len(y) + 1

        if not ((type(percentiles) is list) or (type(percentiles) is np.ndarray)):
            return librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=percentiles, hop_length=hop_len)[0]

        rolloff_freq = np.zeros((len(percentiles), n_frames))
        for i, per in enumerate(percentiles):
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=per, hop_length=hop_len)
            rolloff_freq[i, :] = rolloff

    else:
        # Librosa needs the hop_size and the window size in number of points. Thus the conversion
        hop_size_pt = int(hop_size * sr)
        window_size_pt = int(window_size * sr)
        if not ((type(percentiles) is list) or (type(percentiles) is np.ndarray)):
            return librosa.feature.spectral_rolloff(y=y,
                                                    sr=sr,
                                                    roll_percent=percentiles,
                                                    hop_length=hop_size_pt,
                                                    n_fft=window_size_pt)[0]

        rolloff_size = int(len(y) // hop_size_pt + 1)  # number of frames in that case
        if verbose:
            print("size of the rollof frequencies vector: {}".format(rolloff_size))
        rolloff_freq = np.zeros((len(percentiles), rolloff_size))
        for i, per in enumerate(percentiles):
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=per,
                                                       hop_length=hop_size_pt, n_fft=window_size_pt)
            rolloff_freq[i, :] = rolloff

    return rolloff_freq


def parser_rolloff(row,
                  data_directory_path,
                  n_frames=None,
                  hop_size=15e-3,
                  window_size=25e-3,
                  percentiles=[0.25, 0.5, 0.75],
                  pickled_audio=False,
                  trim=False,
                  db_cutoff=3):
    """
    A version of the roll-off frequencies parser that can be applied to a pandas array. The user can either specify a
    number of frames to divide the signal in OR manually set the hop size and the window size (but not both). By default, the hop_size is set.
    :param row: the row on which the function has to be applied
    :param data_directory_path: the directory of the data
    :param n_frames: the number of frames the audio file is expected to be divided in.
    :param hop_size: the hop size in seconds
    :param window_size: the window size in seconds
    :param percentiles: the percentiles to compute
    :param pickled_audio: specify whether or not the pickled audio file is targeted
    :return: the rollof frequencies as an array
    """
    file_name = osp.join(data_directory_path, row['fname'])
    try:
        # here kaiser_fast is a technique used for faster extraction
        if not pickled_audio:
            X, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        else:
            X, sample_rate = load_pickled_audio(row['fname'], data_directory_path)
        # we extract mfcc feature from data
        
        if trim:
            X = librosa.effects.trim(X, top_db=db_cutoff)[0]
        rolloff_freq = parser_rolloff_raw(y=X, sr=sample_rate, n_frames=n_frames,
                                        percentiles=percentiles, hop_size=hop_size, window_size=window_size)

    except Exception as e:
        print('Error encountered while parsing the file:',file_name)
        return 'None'

    return rolloff_freq


###Extrait du GITHUB : CLASSIFIED

def extract_feature(file_name):
    X, sample_rate = sf.read(file_name, dtype='float32')
    if X.ndim > 1:
        X = X[:,0]
    X = X.T
    # short term fourier transform
    stft = np.abs(librosa.stft(X))
    # mfcc
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    # chroma
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    # melspectrogram
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    # spectral contrast
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs, chroma, mel, contrast, tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext='*.ogg'):
    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
                mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
                print("[Error] extract feature error. %s" % (e))
                continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features])
            # labels = np.append(labels, fn.split('/')[1])
            labels = np.append(labels, label)
        print("extract %s features done" % (sub_dir))
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

# Get features and labels
# r = os.listdir("data/")
# r.sort()
# features, labels = parse_audio_files('data', r)
# np.save('feat.npy', features)
# np.save('label.npy', labels)
### Fin du GITHUB




