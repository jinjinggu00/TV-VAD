import numpy as np

file_path = 'A.Beautiful.Mind.2001__#00-01-45_00-02-50_label_A__0.npy'

data = np.load(file_path, allow_pickle=True)

print(data.shape)
print(data)