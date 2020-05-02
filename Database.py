import numpy as np
import glob
from scipy.io import wavfile
import pandas as pd
from pathlib import Path
import math


def sum_labels(labels):
    """
    :param labels: path to list of reference labels that are not only every 10ms
    :return: New list of labels in form of a dataframe - (averaged over the time window)
    """
    data = pd.read_csv(labels, sep='     ', names=['Time', 'F0'])
    data = {'Time': data['Time'], 'F0': data['F0'], 'Name': [Path(labels).stem[:-3]]*len(data.index)}

    return pd.DataFrame(data, columns=['Time', 'F0', 'Name'])

def read_wav(wav_path):
    """
    :param wav_path: Path to one wav-file
    :return: Pandas Dataframe with normalized data
    """
    fs, audio_data = wavfile.read(wav_path)

    audio_data_proc_mean = audio_data - audio_data.mean()
    audio_data_proc = audio_data_proc_mean / audio_data_proc_mean.std()

    df_audio_data = pd.DataFrame({'Name': [Path(wav_path).stem]*len(audio_data_proc), 'Audio_Data': audio_data_proc})

    return df_audio_data

labels = glob.glob("/home/jonny/Desktop/Trainingsdatenbank/adc2004/*.txt")
wav_path = glob.glob("/home/jonny/Desktop/Trainingsdatenbank/adc2004/*.wav")

label_dataframe = pd.DataFrame(columns=['Time', 'F0', 'Name'])
audio_dataframe = pd.DataFrame(columns=['Name', 'Audio_Data'])


for i in range(len(labels)):

    label_new = Path(wav_path[i]).stem
    label = "/home/jonny/Desktop/Trainingsdatenbank/adc2004/{}REF.txt".format(label_new)
    new_frame = sum_labels(label)
    label_dataframe = label_dataframe.append(new_frame)

    new_audioframe = read_wav(wav_path[i])
    audio_dataframe = audio_dataframe.append(new_audioframe)

    print("Iteration {} of {}".format(i+1, len(labels)))

label_dataframe.to_csv('/home/jonny/Desktop/Trainingsdatenbank/training_label_ADC2004.csv', index=False)
audio_dataframe.to_csv('/home/jonny/Desktop/Trainingsdatenbank/training_data_ADC2004.csv', index=False)

print("Saving Successful")

x = 1


