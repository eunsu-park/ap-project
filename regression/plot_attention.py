import matplotlib.pyplot as plt
import numpy as np

# f = '/Volumes/work/NJIT/01_AP/07_Attention/SINGLE_1_1_04/epoch_0080/attention_batch_0000.npz'
# f = '/Volumes/work/NJIT/01_AP/07_Attention/SINGLE_3_2_02/epoch_0060/attention_batch_0000.npz'
f = '/Volumes/work/NJIT/01_AP/07_Attention/SINGLE_4_3_05/epoch_0080/attention_batch_0000.npz'
data = np.load(f, allow_pickle=True)

# ✅ 메타데이터에서 자동으로 읽기
metadata = data['metadata'].item()
input_size = metadata['seq_len']  # 자동으로 24
target_size = metadata['n_targets']  # 자동으로 8

print(f"Input size: {input_size}")
print(f"Target size: {target_size}")

attn = data['attention_weights'][-1].mean(axis=0)  # Last layer, avg over heads

# ✅ attn.shape으로 검증
actual_seq_len = attn.shape[0]
assert actual_seq_len == input_size, f"Mismatch! attn.shape={attn.shape}, expected {input_size}"

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Attention Matrix
im1 = axes[0].imshow(attn, cmap='hot', aspect='auto')
axes[0].set_xlabel('Key Position (Past)', fontsize=12)
axes[0].set_ylabel('Query Position (Current)', fontsize=12)
axes[0].set_title('Attention Matrix', fontsize=14, fontweight='bold')
plt.colorbar(im1, ax=axes[0])

# 2. Incoming Attention
incoming = attn.sum(axis=0)
# ✅ incoming.shape을 직접 사용 (가장 안전)
axes[1].bar(range(len(incoming)), incoming, color='orangered')
axes[1].set_xlabel('Timestep', fontsize=12)
axes[1].set_ylabel('Total Incoming Attention', fontsize=12)
axes[1].set_title('Temporal Importance', fontsize=14, fontweight='bold')

# ✅ Peak 자동 감지
peak_idx = incoming.argmax()
axes[1].axvline(peak_idx, color='yellow', linestyle='--', linewidth=2, 
                label=f'Peak (t={peak_idx})')
axes[1].legend()

# 3. Attention Distribution
axes[2].plot(attn.T, alpha=0.3)
axes[2].plot(attn.mean(axis=0), 'k-', linewidth=3, label='Average')
axes[2].set_xlabel('Key Position', fontsize=12)
axes[2].set_ylabel('Attention Weight', fontsize=12)
axes[2].set_title('Attention Distribution', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('attention_comprehensive.png', dpi=300, bbox_inches='tight')

data.close()