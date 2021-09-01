from random import shuffle
import numpy as np
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as T
import matplotlib as mpl
import matplotlib.pyplot as plt
from utils import *
import os

learning_rate = 0.1    # Optimization learning rate
FMAP = 130             # Feature map of interest
model_depth = 29       # Layer to obtain features from
iterations = 50        # Max iterations for optimization process

# Initialize model and three-channel noise image
vgg = models.vgg16(pretrained=True, progress=True)
tensor = torch.rand((1, 3, 256, 256), requires_grad=True)

for param in vgg.parameters():
    param.requires_grad = False

# Initialize optimizer (SGD, Adam, ...)
optim = torch.optim.SGD([tensor], lr=learning_rate, weight_decay=1e-4)

# Keep track of intermediate optimization results
frames = []

# Optimization loop
for i in range(iterations+1):
    frames.append(tensor[0].permute(1, 2, 0).detach().numpy().copy())
    optim.zero_grad()
    result = vgg.features[:model_depth](tensor)
    loss = -result[:, FMAP, :, :].sum()    # Since optimizers can only minimize tensor, we optimize with respect to negative loss
    loss.backward(retain_graph=True)
    optim.step()

# Normalize frames so they can be displayed sequentially
min_ = -5
max_ = 5
for i, f in enumerate(frames):
    f = np.clip(f, min_, max_)
    f = (((f - min_) / (max_ - min_)) * 255).astype(np.uint8)
    cp = f.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(f, 'iter={}'.format(i), (210, 240), font, 0.3, (0, 0, 0), 1, cv2.LINE_AA)
    frames[i] = f

export_video(frames, 'optimization_process.mp4', fps=5)

# Output last frame as final result
numpy = cp
fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(111)
ax.imshow(numpy)
ax.axis('off')
fig.savefig('result.png', format='png', bbox_inches='tight')
plt.close(fig)