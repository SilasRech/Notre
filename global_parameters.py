# Important Path to Set
label_dataframe_Path = '/home/jonny/Desktop/Trainingsdatenbank/training_label_ADC2004.csv'
audio_dataframe_Path = '/home/jonny/Desktop/Trainingsdatenbank/training_data_ADC2004.csv'

label_dataframe_Path1 = '/home/jonny/Desktop/Trainingsdatenbank/training_label2.csv'
audio_dataframe_Path1 = '/home/jonny/Desktop/Trainingsdatenbank/training_data2.csv'

# Feature Extraction Parameters

hop_length = 44100*0.01
win_length = 512
n_fft = 1024

number_features = 10
extraction = 'CQT'


# Neural Network Parameter
input_size = 1000
output_size = 64

classes_to_detect = 120
