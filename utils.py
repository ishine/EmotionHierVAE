import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import matplotlib.pyplot as plt
import librosa.display as display

import os


def makedirs(path):
    try:
        original_umask = os.umask(0)
        os.makedirs(path, mode=0o777, exist_ok=False)
    finally:
        os.umask(original_umask)

def check_recon_mel(recon_mel, pth_SaveMel, number, mode='GT'):
    """
        Save reconstruction log-mel spectrogram.
    """
    fig, ax = plt.subplots()
    img = display.specshow(20 * recon_mel, x_axis='time',
                            y_axis='mel', sr=16000,
                            ax=ax, vmax=0, vmin=-100, cmap='magma')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel-frequency spectrogram')    
    
    if mode != "GT":
        plt.savefig(pth_SaveMel + f"/{mode}_mel_{number}.jpg")
        np.save(pth_SaveMel + f"/{mode}_mel_{number}.npy", recon_mel)
    else:
        plt.savefig(pth_SaveMel + f"/GT_mel.jpg")
        np.save(pth_SaveMel + f"/GT_mel.npy", recon_mel)

    plt.close()


def check_tsne(z, target_id, pth_SaveFig, number):
    """
    ? INPUT:
    - z: (dataset_number, channels)
    - target_id: (dataset_number, )
    """
    from sklearn.manifold import TSNE    
    #print(z.shape, target_id.shape)

    model_tsne = TSNE(n_components = 2)

    transformed = model_tsne.fit_transform(z)

    plt.figure()
    xs = transformed[:, 0]
    ys = transformed[:, 1]
    plt.scatter(xs, ys, c=target_id)

    save_path = pth_SaveFig + f"tsne_{number}.jpg"
    plt.savefig(save_path)
    plt.close()
