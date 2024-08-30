import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import animation

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = 18

# Data
data = {
    "model": [
        "CLIP@224", "OpenCLIP", "DINOv2", 
        "SDim", "SD1.5", "SDXL", 
        "DiT", "SD3", "SD2.1", "CLIP+DINOv2@336",
    ],
    "mme": [
        1449.6378551420569, 1460.28331332533, 1295.467787114846, 1205.331632653061,
        1163.8987595038016, 1212.6928771508603, 901.9988995598239, 843.4338735494197,
        905.2746098439375, 1475.1920768307323
    ],
    "language_align": [0.8179882815, 0.5316308595, 0.471748047, 0.3700683595,
                        0.365673828, 0.357919922, 0.3915234375, 0.3767382815, 0.3384765625, 0.511171875
    ],
    "weighted_per_image": [
        14.3, 16.22, 24.51, 20.9, 22.02, 16.52, 1.91, 3.09, 6.99, 26.08
    ]
}


# data = {
#     "model": [ "OpenCLIP", "DINOv2", 
#         "SDim", "SD1.5", "SDXL", 
#         "DiT", "SD3", "SD2.1", "CLIP+DINOv2@336"
#     ],
#     "mme": [ 63.40206186, 58.50515464, 52.83505155,
#         42.5257732, 43.72852234, 33.67697595, 32.81786942,
#         28.86597938, 65.12027491
#     ],
#     "language_align": [0.524101563, 0.461611328, 0.348525391,
#                         0.351435547, 0.357431641, 0.386757813, 0.374101563, 0.348457031, 0.511787109
#     ],
#     "weighted_per_image": [
#         16.22, 24.51, 20.9, 22.02, 16.52, 1.91, 3.09, 6.99, 26.08
#     ]
# }

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Normalize the columns
df["normalized_mmbench_en"] = (df["mme"] - df["mme"].min()) / (df["mme"].max() - df["mme"].min())
df["normalized_language_align"] = (df["language_align"] - df["language_align"].min()) / (df["language_align"].max() - df["language_align"].min())
df["normalized_weighted_per_image"] = (df["weighted_per_image"] - df["weighted_per_image"].min()) / (df["weighted_per_image"].max() - df["weighted_per_image"].min())

# Polynomial regression (degree 2)
poly = PolynomialFeatures(degree=2)
X = df[["normalized_language_align", "normalized_weighted_per_image"]]
y = df["normalized_mmbench_en"]
X_poly = poly.fit_transform(X)
poly_model = LinearRegression()
poly_model.fit(X_poly, y)
poly_pred = poly_model.predict(X_poly)

# Add the predictions to the DataFrame
df["predicted_mmbench_en_poly"] = poly_pred

# Create the mesh grid for plotting the polynomial surface
x = np.linspace(df["weighted_per_image"].min(), df["weighted_per_image"].max()+2, 50)
y = np.linspace(df["language_align"].min(), df["language_align"].max()+0.1, 50)
x_grid, y_grid = np.meshgrid(x, y)
data2 = {
    "language_align": y_grid.flatten().tolist(),
    "weighted_per_image": x_grid.flatten().tolist(),
}


data3 = {
    "model": [
        "CLIP@336", "SigLIP", "CLIP+DINOv2@224"
    ],
    "mme": [1502.696679, 1424.995098, 1436.419568
    ],
    "language_align": [
        0.790546875, 0.5259375, 0.535527344
    ],
    "weighted_per_image": [
        15.66, 12.89, 23.62
    ]
}

# data3 = {
#     "model": [
#         "CLIP@336", "SigLIP", "CLIP+DINOv2@224"
#     ],
#     "mme": [64.26116838, 61.8556701, 65.72164948
#     ],
#     "language_align": [
#         0.788125, 0.504785156, 0.53671875
#     ],
#     "weighted_per_image": [
#         15.66, 12.89, 23.62
#     ]
# }

df2 = pd.DataFrame(data2)

def calculate_distance(x1, y1, z1, x2, y2, z2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)

df2["normalized_language_align"] = (df2["language_align"] - df["language_align"].min()) / (df["language_align"].max() - df["language_align"].min())
df2["normalized_weighted_per_image"] = (df2["weighted_per_image"] - df["weighted_per_image"].min()) / (df["weighted_per_image"].max() - df["weighted_per_image"].min())

# Create polynomial features for the grid points
X2 = df2[["normalized_language_align", "normalized_weighted_per_image"]]
grid_points_poly = poly.transform(X2)

# Predict the normalized mmbench_en values on the grid
z_pred = poly_model.predict(grid_points_poly)
df2["predicted_mmbench_en_poly"] = z_pred

df3 = pd.DataFrame(data3)
df3["normalized_language_align"] = (df3["language_align"] - df["language_align"].min()) / (df["language_align"].max() - df["language_align"].min())
df3["normalized_weighted_per_image"] = (df3["weighted_per_image"] - df["weighted_per_image"].min()) / (df["weighted_per_image"].max() - df["weighted_per_image"].min())

X3 = df3[["normalized_language_align", "normalized_weighted_per_image"]]
pred_points_poly = poly.transform(X3)

z_pred = poly_model.predict(pred_points_poly)
df3["predicted_mmbench_en_poly"] = z_pred

# Revert normalization for plotting
df["mmbench_en_predicted"] = df["predicted_mmbench_en_poly"] * (df["mme"].max() - df["mme"].min()) + df["mme"].min()
df2["predicted_mmbench_en_poly"] = df2["predicted_mmbench_en_poly"] * (df["mme"].max() - df["mme"].min()) + df["mme"].min()
df3["mmbench_en_predicted"] = df3["predicted_mmbench_en_poly"] * (df["mme"].max() - df["mme"].min()) + df["mme"].min()

# Create the 3D scatter plot with different colors for each model
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis limits
x_min, x_max = df["weighted_per_image"].min(), df["weighted_per_image"].max() + 2
y_min, y_max = df["language_align"].min(), df["language_align"].max() + 0.1
z_min, z_max = df["mme"].min(), df["mme"].max() + 100  # Adjust as needed for better visualization

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_zlim(z_min, z_max)

# Define colors for each model

x_offset_base = 0.3
y_offset_base = 0.02
z_offset_base = 10

def update_surface(frame):
    if frame < 20:
        ax.clear()
        
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        
        ax.scatter(df["weighted_per_image"], df["language_align"], df["mme"], color='tab:orange', label='Ground Truth', s=30)
        ax.scatter(df3["weighted_per_image"], df3["language_align"], df3["mme"], color='tab:orange', s=90, marker='*', label='Ground Truth')
        ax.scatter(df3["weighted_per_image"], df3["language_align"], df3["mmbench_en_predicted"], s=150, marker='*', facecolors='none', edgecolors='r', label='Prediction')
        
        for i, model in enumerate(df["model"]):
            ax.text(df["weighted_per_image"][i], df["language_align"][i], df["mme"][i]+10, model, fontsize=12, color='black')

        # Label the new models
        for i, model in enumerate(df3["model"]):
            ax.text(df3["weighted_per_image"][i], df3["language_align"][i], df3["mme"][i]+35, model, fontsize=12, color='black')
        
        ax.set_xlabel('Correspondance (PCK@0.10)', labelpad=7)
        ax.set_ylabel('Cross-modal Alignment', labelpad=7)
        ax.set_zlabel('MLLM Performance (MME)', labelpad=7)
        
    else:
        ax.clear()  # Clear the previous frame
        
        # Reset the axis limits in each frame to prevent changes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)

        ax.scatter(df["weighted_per_image"], df["language_align"], df["mme"], color='tab:orange', label='Ground Truth', s=30)
        ax.scatter(df3["weighted_per_image"], df3["language_align"], df3["mme"], color='tab:orange', s=90, marker='*', label='Ground Truth')
        ax.scatter(df3["weighted_per_image"], df3["language_align"], df3["mmbench_en_predicted"], s=150, marker='*', facecolors='none', edgecolors='r', label='Prediction')
        
        ax.plot_surface(
            df2["weighted_per_image"].to_numpy().reshape(x_grid.shape)[:frame-20, :frame-20],
            df2["language_align"].to_numpy().reshape(x_grid.shape)[:frame-20, :frame-20],
            df2["predicted_mmbench_en_poly"].to_numpy().reshape(x_grid.shape)[:frame-20, :frame-20],
            color='tab:blue', alpha=0.5
        )
        
        for i, model in enumerate(df["model"]):
            ax.text(df["weighted_per_image"][i], df["language_align"][i], df["mme"][i]+10, model, fontsize=12, color='black')

        # Label the new models
        for i, model in enumerate(df3["model"]):
            ax.text(df3["weighted_per_image"][i], df3["language_align"][i], df3["mme"][i]+35, model, fontsize=12, color='black')
        
        ax.set_xlabel('Correspondance (PCK@0.10)', labelpad=7)
        ax.set_ylabel('Cross-modal Alignment', labelpad=7)
        ax.set_zlabel('MLLM Performance (MME)', labelpad=7)

# Create the animation
ani = animation.FuncAnimation(fig, update_surface, frames=70, interval=100, blit=False)

# Save the animation as an MP4 video
ani.save('animated_curve.mp4', writer='ffmpeg', fps=15, dpi=300)
plt.show()