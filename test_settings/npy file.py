import numpy as np

testsettings = np.load("threshold_options.npy")
np.savetxt("foo.csv", testsettings, delimiter=",")
x=1