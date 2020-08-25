import scipy.io
import numpy as np
import matplotlib.pyplot as plt

data = scipy.io.loadmat("../data/ID1/Sz6.mat")
print(data['EEG'].shape)

plt.figure(figsize=(12, 5))

plt.plot(data['EEG'][:, 0], label='observed states', linewidth=1)
plt.xlabel('time')
plt.ylabel('output')

plt.show()

print(data['EEG'][0])