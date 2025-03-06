#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from datetime import datetime, timezone
import filetype
import hashlib
import numpy as np
import os
import pickle
from PIL import Image, ImageOps
# from skimage.exposure import rescale_intensity
import socket
import torch
from torchvision import transforms, models
from warnings import warn


image_dirs = [
              r'FOBmany230728d',
              r'FOBmany240312d',
              r'FOBmin230728d',
              r'FOBsel230517d',
              r'FOBsel230517dAniso',
              r'Song',
              r'Song230509dSel',
              r'SongSq',
              r'SongSqCrop',
              r'Tsao500',
              r'Tsao1593',
              r'Tsao15901',
              ]

min_dimension = 224
pad_color = 255

checksumming = True
skipping = False

skiplist = []


def has_transparency(img):
    # based on https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
    if img.mode == 'LA':
        img = img.convert('RGBA')
    if img.info.get('transparency', None) is not None:
        return True
    if img.mode == 'P':
        transparent = img.info.get('transparency', -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == 'RGBA':
        if img.getextrema()[3][0] < 255:
            return True
    return False


def get_md5_checksum(file):
    hl_md5 = hashlib.md5()
    with open(file, 'rb') as f:
        for chnk in iter(lambda: f.read(4096), b''):
            hl_md5.update(chnk)
    return hl_md5.hexdigest()


activation = {}
def get_activation(name):
    """
    Create hook for extracting intermediate layer feature values ('unit responses') from a model.
        based on https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
        and https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/14
    """
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def relu_inplace_to_false(module):
    # based on https://stackoverflow.com/questions/74124725/setting-relu-inplace-to-false
    for layer in module._modules.values():
        if isinstance(layer, torch.nn.ReLU):
            layer.inplace = False
        relu_inplace_to_false(layer)


def get_alexnet_unit_responses(activ):
    if activ != {} and activ is not None:
        d = {
            # '00_conv1_conv2d': activ['features.0'].cpu().numpy().squeeze(),
            # '01_conv1_relu': activ['features.1'].cpu().numpy().squeeze(),
            # '02_conv1_maxpool2d': activ['features.2'].cpu().numpy().squeeze(),
            # '03_conv2_conv2d': activ['features.3'].cpu().numpy().squeeze(),
            # '04_conv2_relu': activ['features.4'].cpu().numpy().squeeze(),
            # '05_conv2_maxpool2d': activ['features.5'].cpu().numpy().squeeze(),
            # '06_conv3_conv2d': activ['features.6'].cpu().numpy().squeeze(),
            # '07_conv3_relu': activ['features.7'].cpu().numpy().squeeze(),
            # '08_conv4_conv2d': activ['features.8'].cpu().numpy().squeeze(),
            # '09_conv4_relu': activ['features.9'].cpu().numpy().squeeze(),
            # '10_conv5_conv2d': activ['features.10'].cpu().numpy().squeeze(),
            # '11_conv5_relu': activ['features.11'].cpu().numpy().squeeze(),
            # '12_conv5_maxpool2d': activ['features.12'].cpu().numpy().squeeze(),
            # '13_avgpool': activ['avgpool'].cpu().numpy().squeeze(),
            # '14_fc6_dropout': activ['classifier.0'].cpu().numpy().squeeze(),
            '15_fc6_linear': activ['classifier.1'].cpu().numpy().squeeze(),
            '16_fc6_relu': activ['classifier.2'].cpu().numpy().squeeze(),
            # '17_fc7_dropout': activ['classifier.3'].cpu().numpy().squeeze(),
            '18_fc7_linear': activ['classifier.4'].cpu().numpy().squeeze(),
            '19_fc7_relu': activ['classifier.5'].cpu().numpy().squeeze(),
            '20_fc8_linear': activ['classifier.6'].cpu().numpy().squeeze(),
        }
    else:
        warn('No activations found.')
        d = {}
    return d


def get_generic_model_unit_responses(activ):
    if activ != {} and activ is not None:
        d = {}
        for k in activ.keys():
            d[k] = activ[k].cpu().numpy().squeeze()
    else:
        warn('No activations found.')
        d = {}
    return d


def save_record(savestr='asset_record'):
    save_path = os.path.join(asset_record_path, savestr + '.pkl')
    with open(save_path, 'wb') as pkl_file:
        pickle.dump(assets, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


def save_snapshot(savestr='asset_record_snapshot'):
    now = datetime.now(timezone.utc)
    datetime_str = now.strftime('%Y%m%dd%H%M%StUTC')
    savestr = datetime_str + '_' + savestr
    save_record(savestr=savestr)


system_name = socket.gethostname()
if 'galactica' in system_name.lower():
    base_path = r'/Users/davidh/Data/Freiwald/ImageDatasets'
    stimulus_path = r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets/StimSpace'
    device = 'cpu'
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif 'marmostor' in system_name.lower():
    base_path = r'/marmostor/DavidH/ImageDatasets'
    stimulus_path = r'/FreiwaldSync/MarmoScope/Stimulus/Sets/StimSpace'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'obsidian' in system_name.lower():
    base_path = r'F:\Data\ImageDatasets'
    stimulus_path = base_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'dobbin' in system_name.lower():
    base_path = r'D:\Data\ImageDatasets'
    stimulus_path = base_path
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    base_path = None
    stimulus_path = None
    device = 'cpu'
    warn('Unknown system name, paths not set.')

asset_record_name = 'asset_record.pkl'
asset_record_path = base_path + os.path.sep + 'asset_record'
if not os.path.exists(asset_record_path):
    os.makedirs(asset_record_path)
if os.path.exists(os.path.join(asset_record_path, asset_record_name)):
    with open(os.path.join(asset_record_path, asset_record_name), 'rb') as pickle_file:
        assets = pickle.load(pickle_file)

if 'assets' not in locals():
    assets = {}


# AlexNet model implementation from...
# alexnet: https://pytorch.org/hub/pytorch_vision_alexnet/
# https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py
# https://arxiv.org/pdf/1404.5997
# https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf

hooks = {}
alexnet = torch.hub.load('pytorch/vision:v0.19.1', 'alexnet', weights='AlexNet_Weights.IMAGENET1K_V1')
hooks['alexnet'] = {}
for name, module in alexnet.named_modules():
    if name == '':
        continue
    relu_inplace_to_false(module)
    hooks['alexnet'][name] = module.register_forward_hook(get_activation(name))


image_counter = 0
for image_dir in image_dirs:
    image_path = stimulus_path + os.path.sep + image_dir

    # from Bao et al Tsao 2020 Nature
    #   "Decoding accuracy for 40 images using object spaces built by responses of different layers of AlexNet
    #    ... There are multiple points for each layer because we performed PCA before and after pooling, activation,
    #    and normalization functions. Layer fc6 showed the highest decoding accuracy, motivating our use of the object
    #    space generated by this layer throughout the paper."

    image_files = [f for f in os.listdir(image_path) if filetype.is_image(os.path.join(image_path, f))]

    for i_f in image_files:
        if skipping:
            if i_f in skiplist:
                print('Skipped {}... skiplisted.'.format(i_f))
                continue

        if checksumming:
            image_file_md5 = get_md5_checksum(os.path.join(image_path, i_f))
            i = image_file_md5
        else:
            i = i_f

        # if checksumming and i in assets:
        #     print('Skipped {}... already processed.'.format(i_f))
        #     continue

        if i not in assets:
            assets[i] = {
                'filename': i_f,
                'filepath': os.path.join(image_path, i_f),
                # 'code': image_info[ii]['code'],
                # 'title': image_info[ii]['title'],
                # 'category': image_info[ii]['code_category'],
                # 'categorysub': image_info[ii]['code_subcategory'],
                # 'categoryfull': image_info[ii]['code_category_full'],
                # 'readable_title': image_info[ii]['name'],
                # 'readable_category': image_info[ii]['category'],
                # 'readable_subcategory': image_info[ii]['subcategory'],
                # 'url_info': image_info[ii]['url_info'],
                # 'url_image': image_info[ii]['url_image'],
            }
            if checksumming:
                assets[i]['md5'] = image_file_md5
        else:
            print('Skipped {}... already processed ({}).'.format(i_f, i))
            continue

        if 'image_size' in assets[i]:
            if np.min(assets[i]['image_size']) < min_dimension:
                print('Skipped {}... {} is less than minimum side {} px.'.format(i_f,
                                                                                 np.min(assets[i]['image_size']),
                                                                                 min_dimension))
                continue

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        with Image.open(assets[i]['filepath']) as loaded_image:
            padded = False

            assets[i]['image_size'] = loaded_image.size
            if np.min(assets[i]['image_size']) < min_dimension:
                warn('{} will be upsampled, side size {} less than minimum {} px.'.format(i,
                                                                                          np.min(assets[i]['image_size']),
                                                                                          min_dimension))

            if has_transparency(loaded_image):
                assets[i]['image_transparency'] = True
            else:
                assets[i]['image_transparency'] = False

            loaded_image = loaded_image.convert('RGBA')
            if type(pad_color) == int:
                padding_color = (pad_color,) * 3 + (0,)
            elif type(pad_color) == tuple:
                if len(pad_color) == 3:
                    padding_color = pad_color + (0,)
            if 'padding_color' not in locals():
                warn('Padding color not recognized, using white.')
                padding_color = (255,) * 3 + (0,)

            resize_ratio = np.min((min_dimension / loaded_image.size[0], min_dimension / loaded_image.size[1]))
            if resize_ratio != 1.0:
                loaded_image = loaded_image.resize(np.round(resize_ratio * np.array(loaded_image.size)).astype(int),
                                                   resample=Image.Resampling.BICUBIC)

            if loaded_image.size[0] != loaded_image.size[1]:
                loaded_image = ImageOps.pad(loaded_image,
                                            size=(np.max(loaded_image.size[:2]),) * 2,
                                            color=padding_color)
                padded = True

            if assets[i]['image_transparency'] or padded:
                loaded_image = Image.alpha_composite(Image.new('RGBA', loaded_image.size, (255,) * 3),
                                                     loaded_image)

            image = loaded_image.convert('RGB')

            if 'unit_responses' not in assets[i]:
                assets[i]['unit_responses'] = {}

            if 'alexnet' not in assets[i]['unit_responses']:
                assets[i]['unit_responses']['alexnet'] = {}

                activation = {}
                tensor = preprocess(image)
                input_batch = tensor.unsqueeze(0)
                input_batch = input_batch.to(device)
                alexnet.to(device)
                with torch.no_grad():
                    model_output = alexnet(input_batch)
                assets[i]['unit_responses']['alexnet'] = get_alexnet_unit_responses(activation)

                # for k in assets[i]['unit_responses']['alexnet'].keys():
                #     print('{}: shape = {}, mean = {}'.format(k,
                #                                              assets[i]['unit_responses']['alexnet'][k].shape,
                #                                              np.mean(assets[i]['unit_responses']['alexnet'][k])))
                # del activation, tensor, input_batch, model_output

                # probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
                # print(probabilities)
                # with open(r'F:\Data\ImageDatasets\imagenet_classes.txt', 'r') as f:
                #     categories = [s.strip() for s in f.readlines()]
                # # Show top categories per image
                # top5_prob, top5_catid = torch.topk(probabilities, 5)
                # for ii in range(top5_prob.size(0)):
                #     print(categories[top5_catid[ii]], top5_prob[ii].item())

        if image_counter % 1000 == 0 and image_counter > 0:
            save_snapshot(savestr='asset_record_partial')
        image_counter += 1
    save_record()


import colorsys
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt


fc6 = np.array([assets[i]['unit_responses']['alexnet']['15_fc6_linear'] for i in assets])
fc6relu = np.array([assets[i]['unit_responses']['alexnet']['16_fc6_relu'] for i in assets])
fc7 = np.array([assets[i]['unit_responses']['alexnet']['18_fc7_linear'] for i in assets])
fc7relu = np.array([assets[i]['unit_responses']['alexnet']['19_fc7_relu'] for i in assets])
fc8 = np.array([assets[i]['unit_responses']['alexnet']['20_fc8_linear'] for i in assets])

template = np.array(['Tsao15901', 'Tsao1593', 'Tsao500', 'SongSqCrop', 'SongSq', 'Song', 'FOBsel230517dAniso', 'FOBsel230517d', 'FOBmany230728d'])
sort_by_tmpl = lambda x: (np.where(template == x)[0][0])

stimset_labels = np.array([assets[i]['filename'].split('_')[0] for i in assets])
n_stimsets = len(np.unique(stimset_labels))
zorder_to_labels = {k: v for k, v in enumerate(sorted(np.unique(stimset_labels), key=sort_by_tmpl))}
labels_to_zorder = {v: k for k, v in enumerate(sorted(np.unique(stimset_labels), key=sort_by_tmpl))}
stimset_zorder = np.array([labels_to_zorder[stimset_labels[n]] for n in range(len(stimset_labels))])
stimset_colors = np.array([colorsys.hsv_to_rgb(z / n_stimsets, 1.0, 1.0) for z in stimset_zorder])

pca_fc6_2c = PCA(n_components=2)
pca_fc6_50c = PCA(n_components=50)

pca_fc6_2c_X_r = pca_fc6_2c.fit_transform(fc6)
pca_fc6_50c_X_r = pca_fc6_50c.fit_transform(fc6)

print(pca_fc6_2c.explained_variance_ratio_)
pca_fc6_2c_explvar = pd.DataFrame(
    data=zip(range(1, len(pca_fc6_2c.explained_variance_ratio_) + 1),
             pca_fc6_2c.explained_variance_ratio_,
             pca_fc6_2c.explained_variance_ratio_.cumsum()),
    columns=['PCA', 'Explained Variance (%)', 'Total Explained Variance (%)']
    ).set_index('PCA').mul(100).round(1)
print(pca_fc6_2c_explvar)

print(pca_fc6_50c.explained_variance_ratio_)
pca_fc6_50c_explvar = pd.DataFrame(
    data=zip(range(1, len(pca_fc6_50c.explained_variance_ratio_) + 1),
             pca_fc6_50c.explained_variance_ratio_,
             pca_fc6_50c.explained_variance_ratio_.cumsum()),
    columns=['PCA', 'Explained Variance (%)', 'Total Explained Variance (%)']
    ).set_index('PCA').mul(100).round(1)
print(pca_fc6_50c_explvar)

fig, ax = plt.subplots(figsize=(8,8))
ax.bar(x=pca_fc6_2c_explvar.index, height=pca_fc6_2c_explvar['Explained Variance (%)'], label='Explained Variance', width=0.9)
ax.plot(pca_fc6_2c_explvar['Total Explained Variance (%)'], label='Total Explained Variance', marker='o')
plt.ylim(0, 100)
plt.ylabel('Explained Variance (%)')
plt.xlabel('PCA')
plt.grid(True, axis='y')
plt.title('Explained Variance')
plt.legend()
plt.show()

fig, ax = plt.subplots(figsize=(8,8))
ax.bar(x=pca_fc6_50c_explvar.index, height=pca_fc6_50c_explvar['Explained Variance (%)'], label='Explained Variance', width=0.9)
ax.plot(pca_fc6_50c_explvar['Total Explained Variance (%)'], label='Total Explained Variance', marker='o')
plt.ylim(0, 100)
plt.ylabel('Explained Variance (%)')
plt.xlabel('PCA')
plt.grid(True, axis='y')
plt.title('Explained Variance')
plt.legend()
plt.show()


# # Difficult to interpret loadings from AlexNet features
# pca_fc6_2c_loadings = pd.DataFrame(pca_fc6_2c.components_.T, columns=['PC1', 'PC2'])
# import matplotlib.pyplot as plt
# import seaborn as sns
# n_vars = 20
# fig, ax = plt.subplots(figsize=(8,4))
# ax = sns.heatmap(
#     pca_fc6_2c.components_[:, :n_vars],
#     cmap='coolwarm',
#     yticklabels=[f'PC{x}' for x in range(1, pca_fc6_2c.n_components_ + 1)],
#     xticklabels=[f'{x}' for x in range(0, n_vars)],
#     linewidths=1,
#     annot=True,
#     fmt=',.2f',
#     cbar_kws={"shrink": 0.8, "orientation": 'horizontal'}
#     )
# ax.set_aspect("equal")
# plt.title('Loading for Each Variable and Component', weight='bold')
# plt.show()
#
# fig, ax = plt.subplots(figsize=(10, 8))
# plt.scatter(x=pca_fc6_2c_loadings['PC1'], y=pca_fc6_2c_loadings['PC2'])
# plt.axvline(x=0, c="black", label="x=0")
# plt.axhline(y=0, c="black", label="y=0")
# for label, x_val, y_val in zip(pca_fc6_2c_loadings.index, pca_fc6_2c_loadings['PC1'], pca_fc6_2c_loadings['PC2']):  # loadings_fc6['PC1'], loadings_fc6['PC2']):
#     plt.annotate(label, (x_val, y_val), textcoords="offset points", xytext=(0, 10), ha='center')
# plt.title('Visualizing PC1 and PC2 Loadings', weight='bold')
# ax.spines[['right', 'top']].set_visible(False)
# plt.show()

pca_fc6_2c_X_r_df = pd.DataFrame(pca_fc6_2c_X_r, columns=[f'PC{x}' for x in range(1, pca_fc6_2c.n_components_ + 1)])
pca_fc6_50c_X_r_df = pd.DataFrame(pca_fc6_50c_X_r, columns=[f'PC{x}' for x in range(1, pca_fc6_50c.n_components_ + 1)])
# print(pca_fc6_50c_X_r_df.head())

# minfob = os.listdir('/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets/FOBmin/Images/20230728d/')
# newstim = os.listdir('/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets/Chen/short_nonameadd')
#
# image_dotcolors = np.full([n_images, 3], np.nan)
# image_edgecolors = np.full([n_images, 3], np.nan)
# for ii, image_name in enumerate(image_files):
#     if 'Freiwald' in image_name:
#         print('frei: {}'.format(image_name))
#         image_dotcolors[ii] = np.array([1.0, 1.0, 0])
#         if '_Head_' in image_name:
#             image_edgecolors[ii] = np.array([1.0, 0, 0])
#         elif '_Objects_' in image_name:
#             image_edgecolors[ii] = np.array([0, 1.0, 0])
#         elif '_Body_' in image_name:
#             image_edgecolors[ii] = np.array([0, 0, 1.0])
#         else:
#             image_edgecolors[ii] = np.array([0.5, 0.5, 0.5])
#         image_dotcolors[ii] = image_edgecolors[ii]
#     elif 'Song' in image_name:
#         print('song: {}'.format(image_name))
#         image_dotcolors[ii] = np.array([0, 1.0, 1.0])
#         if '_m' in image_name:
#             image_edgecolors[ii] = np.array([1.0, 0, 0])
#         elif '_o' in image_name or 'u' in image_name:
#             image_edgecolors[ii] = np.array([0, 1.0, 0])
#         elif '_b' in image_name:
#             image_edgecolors[ii] = np.array([0, 0, 1.0])
#         else:
#             image_edgecolors[ii] = np.array([0.5, 0.5, 0.5])
#         image_dotcolors[ii] = image_edgecolors[ii]
#     else:
#         print('unknown: {}'.format(image_name))
#         image_dotcolors[ii] = np.array([0.5, 0, 0.5])
#         image_edgecolors[ii] = np.array([0.5, 0, 0.5])
#     # if image_name in minfob:
#     #     image_dotcolors[ii] = np.array([0.5, 0.5, 0.5])
#     if image_name not in newstim:
#         image_dotcolors[ii] = np.array([0.5, 0.5, 0.5])
#         image_edgecolors[ii] = np.array([0.5, 0.5, 0.5])
#
# image_names = [i.replace('.png','').replace('FreiwaldFOB2012_','').replace('FreiwaldFOB2018_','').replace('Song_','').replace('_erode3px','').replace('Objects_','').replace('Head_','') for i in image_files]

fig, ax = plt.subplots(figsize=(10,10))
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})
# ax.scatter(data=pca_fc6_50c_X_r_df, x='PC1', y='PC2', s=40, alpha=0.8)  #, c=image_dotcolors, edgecolors=image_edgecolors, linewidths=2)
# ax.scatter(data=pca_fc6_50c_X_r_df, x='PC1', y='PC2', s=40, alpha=0.8, c=stimset_colors, label=stimset_labels)  # , zorder=stimset_zorder)  # linewidths=2)
for z in np.unique(stimset_zorder):
    if zorder_to_labels[z] == 'Tsao15901' or zorder_to_labels[z] == 'Tsao1593':
        continue
    ax.scatter(data=pca_fc6_50c_X_r_df.iloc[np.where(stimset_zorder==z)], x='PC1', y='PC2', s=10, alpha=0.9, c=stimset_colors[np.where(stimset_zorder==z)], label=zorder_to_labels[z])  # , zorder=stimset_zorder)  # linewidths=2)
# for i, txt in enumerate(image_names):
#     ax.annotate(txt, (pca_fc6_X_r_df.values[i,0], pca_fc6_X_r_df.values[i,1]), horizontalalignment='center')
plt.title('PCA of Stimulus Image AlexNet Features')
plt.legend()
# sns.despine()
fig.show()

fig, ax = plt.subplots(figsize=(10,10))
plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 10})
# ax.scatter(data=pca_fc6_2c_X_r_df, x='PC1', y='PC2', s=40, alpha=0.8)  #, c=image_dotcolors, edgecolors=image_edgecolors, linewidths=2)
# ax.scatter(data=pca_fc6_2c_X_r_df, x='PC1', y='PC2', s=40, alpha=0.8, c=stimset_colors, label=stimset_labels)  # , zorder=stimset_zorder)  # linewidths=2)
for z in np.unique(stimset_zorder):
    if zorder_to_labels[z] == 'Tsao15901' or zorder_to_labels[z] == 'Tsao1593':
        continue
    ax.scatter(data=pca_fc6_2c_X_r_df.iloc[np.where(stimset_zorder==z)], x='PC1', y='PC2', s=10, alpha=0.9, c=stimset_colors[np.where(stimset_zorder==z)], label=zorder_to_labels[z])  # , zorder=stimset_zorder)  # linewidths=2)
# for i, txt in enumerate(image_names):
#     ax.annotate(txt, (pca_fc6_X_r_df.values[i,0], pca_fc6_X_r_df.values[i,1]), horizontalalignment='center')
plt.title('PCA of Stimulus Image AlexNet Features')
plt.legend()
# sns.despine()
fig.show()
