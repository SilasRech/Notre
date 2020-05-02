import numpy as np
import glob
from scipy.io import wavfile
import pandas as pd
from pathlib import Path
from numba import jit
import math


def sum_labels(labels):
    """
    :param labels: path to list of reference labels that are not only every 10ms
    :return: New list of labels in form of a dataframe - (averaged over the time window)
    """
    data = pd.read_csv(labels, sep='\t', names=['F0', 'Time'])
    data = {'Time': data.index, 'F0': data['F0'], 'Name': [Path(labels).stem]*len(data.index)}

    return pd.DataFrame(data, columns=['Time', 'F0', 'Name'])

def read_wav(wav_path):
    """
    :param wav_path: Path to one wav-file
    :return: Pandas Dataframe with normalized data
    """
    fs, audio_data = wavfile.read(wav_path)

    audio_data_proc_mean = audio_data[:, 0] - audio_data[:, 0].mean()
    audio_data_proc = audio_data_proc_mean / audio_data_proc_mean.std()

    df_audio_data = pd.DataFrame({'Name': [Path(wav_path).stem]*len(audio_data_proc), 'Audio_Data': audio_data_proc})

    return df_audio_data

labels = glob.glob("/home/jonny/Desktop/Trainingsdatenbank/MIR-1K_for_MIREX/PitchLabel/*.txt")
wav_path = glob.glob("/home/jonny/Desktop/Trainingsdatenbank/MIR-1K_for_MIREX/Wavfile/*.wav")

label_dataframe = pd.DataFrame(columns=['Time', 'F0', 'Name'])
audio_dataframe = pd.DataFrame(columns=['Name', 'Audio_Data'])

for i in range(len(labels)):

    label_new = Path(wav_path[i]).stem
    label = "/home/jonny/Desktop/Trainingsdatenbank/MIR-1K_for_MIREX/PitchLabel/{}.txt".format(label_new)
    new_frame = sum_labels(label)

    new_audioframe = read_wav(wav_path[i])

    label_dataframe = label_dataframe.append(new_frame)
    audio_dataframe = audio_dataframe.append(new_audioframe)

    print("Iteration {} of {}".format(i+1, len(labels)))


label_dataframe.to_csv('/home/jonny/Desktop/Trainingsdatenbank/training_label2.csv', index=False)
audio_dataframe.to_csv('/home/jonny/Desktop/Trainingsdatenbank/training_data2.csv', index=False)

print("Saving Successful")

x = 1