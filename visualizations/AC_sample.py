import numpy as np
import matplotlib.pyplot as plt
from validate_run import validate_run
import pandas as pd
from tqdm import tqdm
import matplotlib as mpl

# Set the global font to be Times New Roman, applies to all text elements
mpl.rc('font', family='Times New Roman', size=28)

# Function to determine which level 1 region a point belongs to
def AC_sample():
    def determine_region(x, y, level=1):
        region_size = 1 / (2 ** level)
        x_index = int(x // region_size)
        y_index = int(y // region_size)

        # Adjust for edge cases where scores are exactly 1
        if x == 1:
            x_index = (2 ** level) - 1  # Maximum index for the level
        if y == 1:
            y_index = (2 ** level) - 1  # Maximum index for the level

        return (x_index, y_index)

    # Create a dictionary to store points by region
    all_sampled = []
    level = 1
    df = pd.read_csv('/Users/shijiayang/Desktop/Vision_Feature_AC_private/visualizations/AC_score.csv')
    df = df.set_index('Models')
    optimal = {"mmbench_en": "CLIP224+DINOv2", "mme": "CLIP336", "mmmu_val": "OpenCLIP", "ok_vqa": "CLIP336+DINOv2", "textvqa_val": "CLIP336+DINOv2", "vizwiz_vqa_val": "CLIP336", "scienceqa_img": "CLIP336", "seed_image": "CLIP336+DINOv2"}
    benchmark = 'seed_image'
    pred_models = []

    while len(all_sampled) < 13:
        regions = {}
        for model, rows in df.iterrows():
            region_key = determine_region(rows[f'{benchmark}_A'], rows[f'{benchmark}_C'], level)
            if region_key not in regions:
                regions[region_key] = []
            regions[region_key].append((model, rows[f'{benchmark}_A'], rows[f'{benchmark}_C']))

        del_list = []
        for model in all_sampled:
            for k, v in regions.items():
                for m in v:
                    if model == m[0]:
                        del_list.append(k)
        for k in del_list:
            del regions[k]

        # Sample one point from each level 1 region
        remaining_regions = set(regions.keys())

        # Selecting one random region and one random point within that region
        while remaining_regions:
            chosen_region = list(remaining_regions)[np.random.choice(np.arange(len(remaining_regions)))]
            chosen_point = list(regions[chosen_region])[np.random.choice(np.arange(len(regions[chosen_region])))]
            all_sampled.append(chosen_point[0])
            if len(all_sampled) >= 2:
                result, pred_model = validate_run(benchmark, all_sampled, top=3)
                pred_models.append(pred_model)
                if result:
                    return len(all_sampled)
            else:
                if optimal[benchmark] in all_sampled:
                    return len(all_sampled)
            remaining_regions.remove(chosen_region)

        level += 1
    return 13

def random_sample():
    all_sampled = []
    benchmark = 'seed_image'
    all_models = ["CLIP336", "CLIP224", "OpenCLIP", "DINOv2", "SDim", "SD1.5", "SDXL", "DiT", "SD3", "SD2.1", "SigLIP", "CLIP224+DINOv2", "CLIP336+DINOv2"]
    optimal = {"mmbench_en": "CLIP224+DINOv2", "mme": "CLIP336", "mmmu_val": "OpenCLIP", "ok_vqa": "CLIP336+DINOv2", "textvqa_val": "CLIP336+DINOv2", "vizwiz_vqa_val": "CLIP336", "scienceqa_img": "CLIP336", "seed_image": "CLIP336+DINOv2"}
    while len(all_sampled) < 13:
        pred_model = all_models[np.random.choice(np.arange(len(all_models)))]
        all_sampled.append(pred_model)
        if pred_model == optimal[benchmark]:
            return len(all_sampled)
        all_models.remove(pred_model)
    return 13

num_trials = 1000  # You can adjust this number based on your needs
results_random = [random_sample() for _ in tqdm(range(num_trials), desc='Simulating Trials')]
results_AC = [AC_sample() for _ in tqdm(range(num_trials), desc='Simulating Trials')]

bins = range(1, max(max(results_AC), max(results_random)) + 2)
counts_AC, bin_edges_AC = np.histogram(results_AC, bins=bins, density=False)
counts_random, bin_edges_random = np.histogram(results_random, bins=bins, density=False)

# Print the number of samples in each bin
print("AC Predict Histogram Counts:")
for bin_edge, count in zip(bin_edges_AC[:-1], counts_AC):
    print(f"Bin {bin_edge}: {count} samples")

print("\nRandom Highest Histogram Counts:")
for bin_edge, count in zip(bin_edges_random[:-1], counts_random):
    print(f"Bin {bin_edge}: {count} samples")

# Plotting the PDF
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Histogram for AC Predict
plt.hist(results_AC, bins=range(1, max(results_AC) + 2), density=True, label='AC Predict', alpha=0.75, edgecolor='black', cumulative=True)

# Line graph for Random Highest
bin_centers = 0.5 * (bin_edges_random[:-1] + bin_edges_random[1:])
plt.plot(bin_centers, np.cumsum(counts_random) / num_trials, label='Random Highest', color='orange', marker='o', markersize=14)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(True)
ax.spines['bottom'].set_visible(True)

plt.xticks(range(1, max(results_random) + 1))
# plt.legend()
plt.show()
    # print(f"Run Failed for {benchmark}. Optimal cannot be predicted!")
    # print(all_sampled)
    # print(pred_models)
