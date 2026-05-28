import matplotlib.pyplot as plt
import numpy as np

# Data
metrics = ['UTMOS', 'CER', 'WER', 'dCER', 'dWER']

# Resynth data
libriTTS_resynth = [3.45, 3.62, 8.97, 2.44, 5.13]
lj_resynth = [3.87, 3.38, 5.82, 0.78, 2.04]

# TTS data (only LJ Speech available)
lj_tts = [3.76, 6.17, 10.36, 4.13, 7.90]

x = np.arange(len(metrics))  # the label locations
width = 0.35  # the width of the bars

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Subplot 1: Resynth ---
rects1 = ax1.bar(x - width/2, libriTTS_resynth, width, label='resynth_libriTTS', color='skyblue')
rects2 = ax1.bar(x + width/2, lj_resynth, width, label='resynth_lj', color='lightcoral')

ax1.set_ylabel('Mean Values')
ax1.set_title('Resynth: Mean Values of Evaluation Metrics')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.bar_label(rects1, padding=3, fmt='%.2f')
ax1.bar_label(rects2, padding=3, fmt='%.2f')

# --- Subplot 2: TTS ---
# For TTS we only have lj data, so we can just center the bars
rects3 = ax2.bar(x, lj_tts, width, label='tts_lj', color='lightgreen')

ax2.set_ylabel('Mean Values')
ax2.set_title('TTS: Mean Values of Evaluation Metrics')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics)
ax2.legend()
ax2.bar_label(rects3, padding=3, fmt='%.2f')

fig.tight_layout()

plt.savefig('/Users/software/Research/hybrid_tts/metrics_artem/metrics_plot_subplots.png', dpi=300)
print("Plot saved to metrics_plot_subplots.png")
