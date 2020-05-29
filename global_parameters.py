# Important Path to Set
loaded = 0# 0 = is not loaded
loaded_database = 'Benjamin'

if loaded_database == 'Benjamin':
    label_dataframe_Path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/training_label_Benjamin.csv'
    audio_dataframe_Path = 'D:/Backup/Trainingsdatenbank/BenjaminDaten/training_data_Benjamin.csv'
    sample_rate = 16000

elif loaded_database == 'ChineseKaraoke':
    label_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_label2.csv'
    audio_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_data2.csv'
    sample_rate = 16000

else:
    label_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_label_ADC2004.csv'
    audio_dataframe_Path = 'D:/Backup/Trainingsdatenbank/training_data_ADC2004.csv'
    sample_rate = 44100

# Feature Extraction Parameters

hop_length = sample_rate*0.01
win_length = 512
n_fft = 1024

number_features = 16
number_bins = 168
extraction = 'CQT'


# Neural Network Parameter
input_size = 1000
output_size = 64

classes_to_detect = 120
number_units_LSTM = 120
last_filter_size = 128

if extraction == 'FFT':
    input_shape = (win_length+1, number_features, 1)

    label_test_labels = 'D:/Backup/Trainingsdatenbank/train_features/test_label_FFT_first_.npy'
    label_test_data = 'D:/Backup/Trainingsdatenbank/train_features/test_data_FFT_first_.npy'
    label_train_labels = 'D:/Backup/Trainingsdatenbank/train_features/train_label_FFT_first_.npy'
    label_train_data = 'D:/Backup/Trainingsdatenbank/train_features/train_data_FFT_first_.npy'
    label_eval_labels = 'D:/Backup/Trainingsdatenbank/train_features/eval_label_FFT_first_.npy'
    label_eval_data = 'D:/Backup/Trainingsdatenbank/train_features/eval_data_FFT_first_.npy'


else:
    input_shape = (number_bins, number_features, 1)

    # Name Files in the fashion of FeatureExtraction_NumberFeatures
    label_test_labels = 'D:/Backup/Trainingsdatenbank/train_features/test_label_CQT_16_Benjamin.npy'
    label_test_data = 'D:/Backup/Trainingsdatenbank/train_features/test_data_CQT_16_Benjamin.npy'
    label_train_labels = 'D:/Backup/Trainingsdatenbank/train_features/train_labelCQT_16_Benjamin.npy'
    label_train_data = 'D:/Backup/Trainingsdatenbank/train_features/train_data_CQT_16_Benjamin.npy'
    label_eval_labels = 'D:/Backup/Trainingsdatenbank/train_features/eval_label_CQT_16_Benjamin.npy'
    label_eval_data = 'D:/Backup/Trainingsdatenbank/train_features/eval_data_CQT_16_Benjamin.npy'



    #label_test_labels = 'D:/Backup/Trainingsdatenbank/train_features/test_label_CQT32_echt.npy'
    #label_test_data = 'D:/Backup/Trainingsdatenbank/train_features/test_data_CQT32_echt.npy'
    #label_train_labels = 'D:/Backup/Trainingsdatenbank/train_features/train_labelCQT32_echt.npy'
    #label_train_data = 'D:/Backup/Trainingsdatenbank/train_features/train_data_CQT32_echt.npy'
    #label_eval_labels = 'D:/Backup/Trainingsdatenbank/train_features/eval_label_CQT32_echt.npy'
    #label_eval_data = 'D:/Backup/Trainingsdatenbank/train_features/eval_data_CQT32_echt.npy'
