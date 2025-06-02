import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = '/root/data/TGVS/PoR_results.txt'  # replace with your actual path


# Initialize lists
fragment_sizes = []
summary_portions = []
f1_scores = []

# Read and process every 4 lines
with open(file_path, 'r') as file:
    lines = [line.strip() for line in file if line.strip()]  # remove empty lines

for i in range(0, len(lines), 4):
    try:
        frag_size = int(lines[i].split(':')[-1].strip())
        summary_portion = int(lines[i + 1].split(':')[-1].strip())
        f1_score = float(lines[i + 3].split(':')[-1].strip())

        fragment_sizes.append(frag_size)
        summary_portions.append(summary_portion)
        f1_scores.append(f1_score)
    except Exception as e:
        print(f"Skipping block at line {i} due to error: {e}")
        continue

# Create DataFrame
df = pd.DataFrame({
    'FragmentSize': fragment_sizes,
    'SummaryPortion': summary_portions,
    'F1Score': f1_scores
})

# Pivot for heatmap
heatmap_data = df.pivot(index='FragmentSize', columns='SummaryPortion', values='F1Score')

# Plot heatmap
plt.figure(figsize=(10, 6))

ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="Greens",
    annot_kws={"size": 14}  # 👈 increase number font size inside cells
)

# Set axis tick font sizes
ax.set_xticklabels(ax.get_xticklabels(), fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=14)

# Axis labels
plt.xlabel('Summary Portion (%)', fontsize=20)
plt.ylabel('Fragment Size (%)', fontsize=20)

plt.tight_layout()
plt.savefig('/root/TGVS/Eval/PoR_HeatMap.png')
plt.savefig('/root/TGVS/Eval/PoR_HeatMap.pdf')
plt.close()