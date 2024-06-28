"""borrowed from https://github.com/Tsingularity/dift/blob/main/src/utils/visualization.py"""

import gc
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class Demo:

    def __init__(self, imgs, ft, img_size, dist='argmax'):
        self.ft = ft # NCHW
        self.imgs = imgs
        self.num_imgs = len(imgs)
        self.img_size = img_size
        self.dist = dist

    def plot_img_pairs(self, fig_size=3, alpha=0.45, scatter_size=70):

        fig, axes = plt.subplots(1, self.num_imgs, figsize=(fig_size*self.num_imgs, fig_size))

        plt.tight_layout()

        for i in range(self.num_imgs):
            axes[i].imshow(self.imgs[i])
            axes[i].axis('off')
            if i == 0:
                axes[i].set_title('source image')
            else:
                axes[i].set_title('target image')

        num_channel = self.ft.size(1)
        cos = nn.CosineSimilarity(dim=1)

        def onclick(event):
            if event.inaxes == axes[0]:
                with torch.no_grad():
                    
                    x, y = int(np.round(event.xdata)), int(np.round(event.ydata))

                    src_ft = self.ft[0].unsqueeze(0)
                    up_sample_scale = self.img_size // src_ft.size(2)
                    src_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(src_ft)
                    src_vec = src_ft[0, :, y, x].view(1, num_channel, 1, 1)  # 1, C, 1, 1

                    del src_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    trg_ft = nn.Upsample(size=(self.img_size, self.img_size), mode='bilinear')(self.ft[1:]) # 1, C, H, W
                    if self.dist == 'argmax':
                        cos_map = cos(src_vec, trg_ft).cpu().numpy()    # 1, H, W
                    if self.dist == 'window-masked':
                        cos_map = cos(src_vec, trg_ft).cpu().numpy()
                        H, W = cos_map.shape[1], cos_map.shape[2]
                        cos_map = cos_map.reshape(-1)
                        # get the position with maximum similarity
                        max_idx = np.argmax(cos_map)
                        max_idx_x = max_idx % W
                        max_idx_y = max_idx // W
                        # get the window mask
                        window_mask = np.zeros((H, W))
                        window_left = max_idx_x - 7 * up_sample_scale
                        window_right = max_idx_x + 8 * up_sample_scale
                        window_top = max_idx_y - 7 * up_sample_scale
                        window_bottom = max_idx_y + 8 * up_sample_scale
                        # clip the window
                        if window_left < 0:
                            window_left = 0
                        if window_right > W:
                            window_right = W
                        if window_top < 0:
                            window_top = 0
                        if window_bottom > H:
                            window_bottom = H
                        # set the window mask
                        window_mask[window_top:window_bottom, window_left:window_right] = 1
                        # apply the window mask
                        cos_map = cos_map.reshape(1, H, W)
                        cos_map = cos_map * window_mask
                        
                    del trg_ft
                    gc.collect()
                    torch.cuda.empty_cache()

                    axes[0].clear()
                    axes[0].imshow(self.imgs[0])
                    axes[0].axis('off')
                    axes[0].scatter(x, y, c='r', s=scatter_size)
                    axes[0].set_title('source image')

                    for i in range(1, self.num_imgs):
                        if self.dist == 'argmax':
                            max_yx = np.unravel_index(cos_map[i-1].argmax(), cos_map[i-1].shape)
                        else:
                            pass # TODO: window soft argmax

                        axes[i].clear()

                        heatmap = cos_map[i-1]
                        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))  # Normalize to [0, 1]
                        axes[i].imshow(self.imgs[i])
                        axes[i].imshow(255 * heatmap, alpha=alpha, cmap='viridis')
                        axes[i].axis('off')
                        axes[i].scatter(max_yx[1], max_yx[0], c='r', s=scatter_size)
                        axes[i].set_title('target image')

                    del cos_map
                    del heatmap
                    gc.collect()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
