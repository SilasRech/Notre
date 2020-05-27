import numpy as np
import math


# Task 4.1
def hz_to_mel(x):
    """
    Converts a frequency given in Hz into the corresponding Mel frequency.

    :param x: input frequency in Hz.
    :return: frequency in mel-scale.
    """
    
    mel = 2595 * np.log10(1 + x / 700)
    
    return mel
    

# Task 4.2
def mel_to_hz(x):
    """
    Converts a frequency given in Mel back into the linear frequency domain in Hz.

    :param x: input frequency in mel.
    :return: frequency in Hz.
    """

    hz = (10 ** (x / 2595) - 1) * 700
    
    return hz


def sec_to_samples(x, sampling_rate):
    """
    Converts continuous time to sample index.

    :param x: scalar value representing a point in time in seconds.
    :param sampling_rate: sampling rate in Hz.
    :return: sample_index.
    """
    
    sample_index = int(x* sampling_rate)
    
    return sample_index


def next_pow2(x):
    """
    Returns the next power of two for any given positive number.

    :param x: scalar input number.
    :return: next power of two larger than input number.
    """ 
    
    return math.ceil(math.log(x, 2))


def get_num_frames(signal_length_samples, window_size_samples, hop_size_samples):
    """
    Returns the total number of frames for a given signal length with corresponding window and hop sizes.

    :param signal_length_samples: total number of samples.
    :param window_size_samples: window size in samples.
    :param hop_size_samples: hop size (frame shift) in samples.
    :return: total number of frames.
    """
    o = window_size_samples - hop_size_samples
    
    return math.ceil((signal_length_samples - o)/(window_size_samples - o)) 


