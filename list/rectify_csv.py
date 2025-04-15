import pandas as pd

file_path = './xd_CLIP_rgbtest.csv'
d = pd.read_csv(file_path, header=None)

# Replace the path
d[0] = d[0].str.replace('home/zhangheng/paper_code/VadCLIP-main/VadCLIP-main/data', 'home/zhangheng/TV-VAD/dataset', regex=False)

# Save the modified CSV file
d.to_csv('xd_CLIP_rgbtest.csv', index=False, header=False)

