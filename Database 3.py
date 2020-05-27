import glob
from scipy.io import wavfile
import pandas as pd
from pathlib import Path


def sum_labels(labels):
    """
    :param labels: path to list of reference labels that are not only every 10ms
    :return: New list of labels in form of a dataframe - (averaged over the time window)
    """
    data = pd.read_excel(labels, header=None)

    data1 = {'Time': data.iloc[6:].index*0.01, 'F0_1': data.iloc[6:][1], 'F0_2': data.iloc[6:][2], 'F0_3': data.iloc[6:][3], 'F0_4': data.iloc[6:][4], 'F0_5': data.iloc[6:][5], 'Name': [Path(labels).stem]*(len(data.index)-6)}
    return pd.DataFrame(data1, columns=['Time', 'F0_1', 'F0_2', 'F0_3', 'F0_4', 'F0_5', 'Name'])

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

labels = glob.glob("D:/Backup/Trainingsdatenbank/BenjaminDaten/*.xls")
wav_path = glob.glob("D:/Backup/Trainingsdatenbank/BenjaminDaten/*.wav")

label_dataframe = pd.DataFrame(columns=['Time', 'F0_1', 'F0_2', 'F0_3', 'F0_4', 'F0_5', 'Name'])
audio_dataframe = pd.DataFrame(columns=['Name', 'Audio_Data'])

for i in range(len(labels)):

    label_new = Path(wav_path[i]).stem
    label = "D:/Backup/Trainingsdatenbank/BenjaminDaten/{}_Labels.xls".format(label_new)
    new_frame = sum_labels(label)

    new_audioframe = read_wav(wav_path[i])

    label_dataframe = label_dataframe.append(new_frame)
    audio_dataframe = audio_dataframe.append(new_audioframe)

    print("Iteration {} of {}".format(i+1, len(labels)))


label_dataframe.to_csv('D:/Backup/Trainingsdatenbank/BenjaminDaten/training_label_Benjamin.csv', index=False)
audio_dataframe.to_csv('D:/Backup/Trainingsdatenbank/BenjaminDaten/training_data_Benjamin.csv', index=False)

print("Saving Successful")

x = 1