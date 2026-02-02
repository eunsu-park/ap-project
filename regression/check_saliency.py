import numpy as np
import matplotlib.pyplot as plt



# file_path = "/Volumes/work/NJIT/01_AP/06_Saliency/SINGLE_1_1_04/epoch_0080/ig_all_targets_batch_0000.npz"
# file_path = "/Volumes/work/NJIT/01_AP/06_Saliency/SINGLE_3_2_02/epoch_0060/ig_all_targets_batch_0000.npz"
file_path = "/Volumes/work/NJIT/01_AP/06_Saliency/SINGLE_4_3_05/epoch_0080/ig_all_targets_batch_0000.npz"

npz = np.load(file_path)
data = npz["data"]
ig = npz["ig"]
print(data.shape, ig.shape)
print(data.min(), data.max())
print(ig.min(), ig.max())


channel_idx = 0
datas = np.hstack([data[channel_idx, n] for n in range(data.shape[1])])
plt.imshow(datas)
plt.show()

channel_idx = 0
target_idx = 0
igs = np.hstack([ig[target_idx, channel_idx, n] for n in range(ig.shape[2])])
plt.imshow(igs)
plt.show()


channel_idx = 0
frame_idx = 0
igs = np.hstack([ig[n, channel_idx, frame_idx] for n in range(ig.shape[0])])
plt.imshow(igs)
plt.show()
