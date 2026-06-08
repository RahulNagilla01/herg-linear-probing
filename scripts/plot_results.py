import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import sys
sys.path.append(".")

# Load results
with open("results/metrics.json") as f:
    results = json.load(f)

lr_accs  = results["lr"]
dom_accs = results["dom"]
layers   = list(range(len(lr_accs)))

# ── Plot ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(layers, lr_accs,  marker='o', linewidth=2,
        color='#2196F3', label='Logistic Regression')
ax.plot(layers, dom_accs, marker='s', linewidth=2,
        color='#FF5722', linestyle='--', label='Difference-of-Means')

# Annotate LR peak
peak_layer = int(np.argmax(lr_accs))
peak_val   = lr_accs[peak_layer]
ax.annotate(
    f'Peak: Layer {peak_layer}\nAcc = {peak_val:.3f}',
    xy=(peak_layer, peak_val),
    xytext=(peak_layer + 1.2, peak_val - 0.03),
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=9
)

# Shade gap between methods
ax.fill_between(layers, dom_accs, lr_accs, alpha=0.1,
                color='#2196F3', label='LR–DoM gap')

# Formatting
ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title(
    'Linear Probe Accuracy per Layer — hERG Channel Blocking\n'
    'Pythia-1B | Logistic Regression vs Difference-of-Means',
    fontsize=12
)
ax.set_xticks(layers)
ax.set_ylim(0.45, 0.90)
ax.axhline(0.5, color='gray', linewidth=0.8,
           linestyle=':', label='Random baseline (0.5)')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/probe_accuracy.png', dpi=150)
print("Saved → results/figures/probe_accuracy.png")
plt.show()