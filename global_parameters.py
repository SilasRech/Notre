# Important Path to Set
loaded = 0# 0 = is not loaded
loaded_database = 'Benjamin'

if loaded_database == 'Benjamin':
    label_dataframe_Path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/training_label_Benjamin.csv'
    audio_dataframe_Path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/training_data_Benjamin.csv'
    sample_rate = 44100

elif loaded_database == 'ChineseKaraoke':
    label_dataframe_Path = 'C:/Users/silas/Desktop/trainingData/training_label2.csv'
    audio_dataframe_Path = 'C:/Users/silas/Desktop/trainingData/training_data2.csv'
    sample_rate = 16000

else:
    label_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_label_ADC2004.csv'
    audio_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_data_ADC2004.csv'
    sample_rate = 44100

# Feature Extraction Parameters

hop_length = 44100*0.01
win_length = 512
n_fft = 1024

number_features = 32
extraction = 'CQT'


# Neural Network Parameter
input_size = 1000
output_size = 64

classes_to_detect = 120
number_units_LSTM = 120
last_filter_size = 128

if extraction == 'FFT':
    input_shape = (win_length+1, number_features, 1)

    label_test_labels = 'D:/Backup/Trainingsdatenbank/train_features/test_label_FFT{}.npy'
    label_test_data = 'D:/Backup/Trainingsdatenbank/train_features/test_data_FFT{}.npy'
    label_train_labels = 'D:/Backup/Trainingsdatenbank/train_features/train_label_FFT{}.npy'
    label_train_data = 'D:/Backup/Trainingsdatenbank/train_features/train_data_FFT{}.npy'


else:
    input_shape = (84, number_features, 1)

    label_test_labels = 'D:/Backup/Trainingsdatenbank/train_features/test_label_CQT32_test{}.npy'
    label_test_data = 'D:/Backup/Trainingsdatenbank/train_features/test_data_CQT32{}_test.npy'
    label_train_labels = 'D:/Backup/Trainingsdatenbank/train_features/train_label_CQT32_test{}.npy'
    label_train_data = 'D:/Backup/Trainingsdatenbank/train_features/train_data_CQT32{}_test.npy'
