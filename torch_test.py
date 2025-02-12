# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:45:17 2024

@author: DavidH
"""

# Insights from ...
# https://pytorch.org/hub/pytorch_vision_alexnet/
# https://blog.paperspace.com/alexnet-pytorch/
# https://www.kaggle.com/code/vortexkol/alexnet-cnn-architecture-on-tensorflow-beginner

import filetype
import numpy as np
import os
from PIL import Image, ImageOps
import socket
import torch
from torchvision import transforms

# Create hook for getting intermediate layer output
# based on https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
#     and https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/14
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


# based on https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
def has_transparency(image):
    if image.mode == 'LA':
        image = image.convert('RGBA')
    if image.info.get('transparency', None) is not None:
        return True
    if image.mode == 'P':
        transparent = image.info.get('transparency', -1)
        for _, index in image.getcolors():
            if index == transparent:
                return True
    elif image.mode == 'RGBA':
        if image.getextrema()[3][0] < 255:
            return True
    return False


system_name = socket.gethostname()
if 'Galactica' in system_name or 'galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/'
    stim_path = r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets'
    collated_stim_path = os.path.join(base_path, 'stims')
    device = 'cpu'
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data'
    stim_path = r'F:\Sync\Freiwald\MarmoScope\Stimulus\Sets'
    collated_stim_path = os.path.join(base_path, 'stimuli')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data'
    stim_path = r'C:\Users\DavidH\Sync\Freiwald\MarmoScope\Stimulus\Sets'
    collated_stim_path = os.path.join(base_path, 'stims')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    base_path = None
    stim_path = None
    collated_stim_path = None
    device = 'cpu'

# Read ImageNet categories
with open(os.path.join(base_path, 'imagenet_classes.txt'), 'r') as f:
    categories = [s.strip() for s in f.readlines()]

model = torch.hub.load('pytorch/vision:v0.19.1', 'alexnet', pretrained=True)
fc6 = model.classifier[1]  # output from 'fc6' (1): Linear(in_features=9216, out_features=4096, bias=True)
fc6.register_forward_hook(get_activation('fc6'))
fc6relu = model.classifier[2]
fc6relu.register_forward_hook(get_activation('fc6relu'))
fc7 = model.classifier[4]  # output from 'fc6' (1): Linear(in_features=9216, out_features=4096, bias=True)
fc7.register_forward_hook(get_activation('fc7'))
fc7relu = model.classifier[5]
fc7relu.register_forward_hook(get_activation('fc7relu'))

image_files = [f for f in os.listdir(collated_stim_path)
               if os.path.isfile(os.path.join(collated_stim_path, f))
               and filetype.is_image(os.path.join(collated_stim_path, f))]
image_files.sort()
n_images = len(image_files)

background_intensity = 128
fc6_features = np.full([n_images, 4096], np.nan)
fc6relu_features = np.full([n_images, 4096], np.nan)
fc7_features = np.full([n_images, 4096], np.nan)
fc7relu_features = np.full([n_images, 4096], np.nan)
for i_im, image_file in enumerate(image_files):
    filename = os.path.join(collated_stim_path, image_file)
    with Image.open(filename) as image:
        if has_transparency(image):
            # print(f'{image_file} has transparency')
            image = image.convert('RGBA')
            background = Image.new('RGBA', image.size, (background_intensity,) * 3)
            composite = Image.alpha_composite(background, image)
            image = composite.convert('RGB')
        else:
            # print(f'{image_file} does not have transparency')
            image = image.convert('RGB')

        if image.size[0] != image.size[1]:
            # print(f'{image_file} is not square')
            image = ImageOps.pad(image, size=(np.max(image.size[:2]),) * 2, color=(background_intensity,) * 3)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image)
    input_batch = tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_batch)

    fc6_features[i_im, :] = activation['fc6'].cpu().numpy().squeeze()
    # fc6relu_features[i_im, :] = activation['fc6relu'].cpu().numpy().squeeze()
    fc7_features[i_im, :] = activation['fc7'].cpu().numpy().squeeze()
    # fc7relu_features[i_im, :] = activation['fc7relu'].cpu().numpy().squeeze()

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    print(image_file, categories[top_catid], top_prob.item())


from sklearn.decomposition import PCA

# pca_fc6 = PCA()
# pca_fc6.fit(fc6_features)
pca_fc6 = PCA(n_components=2)
pca_fc6_X_r = pca_fc6.fit_transform(fc6_features)

# pca_fc7 = PCA()
# pca_fc7.fit(fc7_features)
pca_fc7 = PCA(n_components=2)
pca_fc7_X_r = pca_fc7.fit_transform(fc7_features)


import pandas as pd

pca_fc6_explvar = pd.DataFrame(
    data=zip(range(1, len(pca_fc6.explained_variance_ratio_) + 1),
             pca_fc6.explained_variance_ratio_,
             pca_fc6.explained_variance_ratio_.cumsum()),
    columns=['PCA', 'Explained Variance (%)', 'Total Explained Variance (%)']
    ).set_index('PCA').mul(100).round(1)
# print(df_expl_var)

pca_fc7_explvar = pd.DataFrame(
    data=zip(range(1, len(pca_fc7.explained_variance_ratio_) + 1),
             pca_fc7.explained_variance_ratio_,
             pca_fc7.explained_variance_ratio_.cumsum()),
    columns=['PCA', 'Explained Variance (%)', 'Total Explained Variance (%)']
    ).set_index('PCA').mul(100).round(1)


loadings_fc6 = pca_fc6.components_

import matplotlib.pyplot as plt
import seaborn as sns
fig, ax = plt.subplots(figsize=(8,8))

ax = sns.heatmap(
    pca_fc6.components_,
    cmap='coolwarm',
    yticklabels=[f'PCA{x}' for x in range(1, pca_fc6.n_components_ + 1)],
    xticklabels=list(pca_fc6_explvar.columns),
    linewidths=1,
    annot=True,
    fmt=',.2f',
    cbar_kws={"shrink": 0.8, "orientation": 'horizontal'}
    )
ax.set_aspect("equal")
plt.title('Loading for Each Variable and Component', weight='bold')
plt.show()


loadings_fc6_pc1 = loadings_fc6[0]
loadings_fc6_pc2 = loadings_fc6[1]
fig, ax = plt.subplots(figsize=(10, 8))
plt.scatter(
    x=loadings_fc6_pc1,  # loadings_fc6['PC1'],
    y=loadings_fc6_pc2,  # loadings_fc6['PC2'],
)

plt.axvline(x=0, c="black", label="x=0")
plt.axhline(y=0, c="black", label="y=0")

for label, x_val, y_val in zip(range(len(loadings_fc6)), loadings_fc6_pc1, loadings_fc6_pc2):  # loadings_fc6['PC1'], loadings_fc6['PC2']):
    plt.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(0, 10), ha='center')

plt.title('Visualizing PCA1 and PCA2 Loadings', weight='bold')
ax.spines[['right', 'top', ]].set_visible(False)
plt.show()



pca_fc6_X_r_df = pd.DataFrame(pca_fc6_X_r, columns=['PCA1', 'PCA2'])  # , index=df.index)
print(pca_fc6_X_r_df.head())


minfob = os.listdir('/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets/FOBmin/Images/20230728d/')
newstim = os.listdir('/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets/Chen/short_nonameadd')

image_dotcolors = np.full([n_images, 3], np.nan)
image_edgecolors = np.full([n_images, 3], np.nan)
for ii, image_name in enumerate(image_files):
    if 'Freiwald' in image_name:
        print('frei: {}'.format(image_name))
        image_dotcolors[ii] = np.array([1.0, 1.0, 0])
        if '_Head_' in image_name:
            image_edgecolors[ii] = np.array([1.0, 0, 0])
        elif '_Objects_' in image_name:
            image_edgecolors[ii] = np.array([0, 1.0, 0])
        elif '_Body_' in image_name:
            image_edgecolors[ii] = np.array([0, 0, 1.0])
        else:
            image_edgecolors[ii] = np.array([0.5, 0.5, 0.5])
    elif 'Song' in image_name:
        print('song: {}'.format(image_name))
        image_dotcolors[ii] = np.array([0, 1.0, 1.0])
        if '_m' in image_name:
            image_edgecolors[ii] = np.array([1.0, 0, 0])
        elif '_o' in image_name or 'u' in image_name:
            image_edgecolors[ii] = np.array([0, 1.0, 0])
        elif '_b' in image_name:
            image_edgecolors[ii] = np.array([0, 0, 1.0])
        else:
            image_edgecolors[ii] = np.array([0.5, 0.5, 0.5])
    else:
        print('unknown: {}'.format(image_name))
        image_dotcolors[ii] = np.array([0.5, 0, 0.5])
        image_edgecolors[ii] = np.array([0.5, 0, 0.5])
    if image_name in minfob:
        image_dotcolors[ii] = np.array([1.0, 0, 1.0])
    if image_name in newstim:
        image_dotcolors[ii] = np.array([0.4, 0.4, 0.4])

image_names = [i.replace('.png','').replace('FreiwaldFOB2012_','').replace('FreiwaldFOB2018_','').replace('Song_','').replace('_erode3px','').replace('Objects_','').replace('Head_','') for i in image_files]

fig, ax = plt.subplots(figsize=(10,10))
plt.rcParams['figure.dpi'] = 600
plt.rcParams.update({'font.size': 1})
# ax.scatter(data=pca_fc6_X_r_df, x='PCA1', y='PCA2', s=20, alpha=0.5, c=np.random.rand(len(pca_fc6_X_r_df), 3))
# ax.scatter(data=pca_fc6_X_r_df, x='PCA1', y='PCA2', s=20, alpha=0.5, c=image_dotcolors)
ax.scatter(data=pca_fc6_X_r_df, x='PCA1', y='PCA2', s=40, alpha=1.0, c=image_dotcolors, edgecolors=image_edgecolors, linewidths=2)
for i, txt in enumerate(image_names):
    ax.annotate(txt, (pca_fc6_X_r_df.values[i,0], pca_fc6_X_r_df.values[i,1]), horizontalalignment='center')
# plt.title('Visualizing Original Data Follow PCA')
plt.title('PC1-2 for Stimulus Image AlexNet Features')
# sns.despine()
fig.show()

# fig, ax = plt.subplots(figsize=(10,10))
# sns.scatterplot(data=pca_fc6_X_r_df, x='PCA1', y='PCA2', ax=ax, s=100, alpha=0.5)
# plt.title('Visualizing Original Data Follow PCA')
# sns.despine()
# fig.show()


# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename).convert('RGBA')
# newbg = Image.new('RGBA', input_image.size, (128, 128, 128))
# ac = Image.alpha_composite(newbg, input_image)
# input_image = ac.convert('RGB')
# preprocess = transforms.Compose([
#     # transforms.Resize(256),
#     # transforms.Resize(224),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# transforms.functional.to_pil_image(input_tensor)

#
# # move the input and model to GPU for speed if available
# if torch.cuda.is_available():
#     input_batch = input_batch.to('cuda')
#     model.to('cuda')
#
# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# print(probabilities)

# # Download ImageNet labels
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# # Read the categories
# with open(r"F:\Data\stimuli\imagenet_classes.txt", "r") as f:
#     categories = [s.strip() for s in f.readlines()]
# # Show top categories per image
# top5_prob, top5_catid = torch.topk(probabilities, 5)
# for i in range(top5_prob.size(0)):
#     print(categories[top5_catid[i]], top5_prob[i].item())
