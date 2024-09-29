#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import filetype
import hashlib
import numpy as np
import os
import pickle
from PIL import Image, ImageOps
import socket
import torch
from torchvision import transforms, models
from warnings import warn


image_info_file = r'20240908d203128tUTC_stickpng_image_info_full.pickle'
background_intensity = 255

checksumming = True
skipping = False

# url_base = 'https://www.stickpng.com/'
# url_cat = url_base + 'cat/'
# url_assets = 'https://assets.stickpng.com/images/'

skiplist = [
    'animals/bilbies',
    'animals/birds/bird-silhouettes',
    'bots-and-robots',
    'cartoons',
    'comics-and-fantasy',
    'icons-logos-emojis',
    'holidays',
    'memes',
    'religion',
    'sports/cricket-teams',
    'sports/ice-hockey/american-hockey-league',
    'sports/ice-hockey/asia-league-ice-hockey',
    'sports/ice-hockey/australian-ice-hockey-league',
    'sports/ice-hockey/belgian-ice-hockey-teams',
    'sports/ice-hockey/champions-hockey-league',
    'sports/ice-hockey/eastern-hockey-league',
    'sports/ice-hockey/echl',
    'sports/ice-hockey/elite-ice-hockey-league',
    'sports/ice-hockey/federal-hockey-league',
    'sports/ice-hockey/french-ice-hockey-teams',
    'sports/ice-hockey/german-ice-hockey-teams',
    'sports/ice-hockey/international-ice-hockey-teams',
    'sports/ice-hockey/kontinental-hockey-league',
    'sports/ice-hockey/ligue-magnus',
    'sports/ice-hockey/ligue-nordamericaine-de-hockey',
    'sports/ice-hockey/national-hockey-league',
    'sports/ice-hockey/national-ice-hockey-league',
    'sports/ice-hockey/ontario-hockey-league',
    'sports/ice-hockey/quebec-major-junior-hockey-league',
    'sports/ice-hockey/southern-professional-hockey-league',
    'sports/ice-hockey/united-states-hockey-league',
    'sports/ice-hockey/us-premier-hockey-league',
    'sports/ice-hockey/western-hockey-league',
    'sports/baseball/major-league-baseball-mlb',
    'sports/rugby-teams-scotland',
    'sports/nfl-football',
]


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


# Create hook for extracting intermediate layer output
# based on https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
#     and https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/14
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


system_name = socket.gethostname()
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/ImageDatasets/stickpng'
    asset_path = base_path + os.path.sep + 'assets'
    tree_path = base_path + os.path.sep + 'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cpu'
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif 'marmostor' in system_name:
    base_path = r'/marmostor/DavidH/ImageDatasets/stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data\ImageDatasets\stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data\ImageDatasets\stickpng'
    asset_path = base_path + os.path.sep + r'assets'
    tree_path = base_path + os.path.sep + r'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    base_path = None
    asset_path = None
    tree_path = None
    infosave_path = None
    device = 'cpu'
    warn('Unknown system name, paths not set.')


with open(os.path.join(infosave_path, image_info_file), 'rb') as file:
    loaded_object = pickle.load(file)
image_info = loaded_object[0]


# alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights='AlexNet_Weights.DEFAULT')


assets = {}
# ii = list(image_info.keys())[0]
for ii in image_info:
    if image_info[ii]['code_category_full'] is None or image_info[ii]['code_category'] is None:
        warn('No category found for {}, skipping...'.format(ii))
        continue
    if skipping:
        if np.any([skiplist[s] in image_info[ii]['code_category_full']
                   for s, _ in enumerate(skiplist)]):
            print('Blacklisted category or subcategory ({}) '
                  'for {}, skipping...'.format(image_info[ii]['code_category_full'], ii))
            continue

    if ii not in assets:
        assets[ii] = {
            'code': image_info[ii]['code'],
            'title': image_info[ii]['title'],
            'category': image_info[ii]['code_category'],
            'subcategory': image_info[ii]['code_subcategory'],
            'fullcategory': image_info[ii]['code_category_full'],
            'readable_title': image_info[ii]['name'],
            'readable_category': image_info[ii]['category'],
            'readable_subcategory': image_info[ii]['subcategory'],
            'url_info': image_info[ii]['url_info'],
            'url_image': image_info[ii]['url_image'],
        }

    image_path = os.path.join(asset_path, image_info[ii]['code'] + '.png')

    if checksumming:
        calculate_sha256 = False
        calculate_md5 = False
        if ii in assets:
            if 'checksum_sha256' not in assets[ii]:
                calculate_sha256 = True
            if 'checksum_md5' not in assets[ii]:
                calculate_md5 = True
        h_sha256 = hashlib.sha256()
        h_md5 = hashlib.md5()
        with open(image_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                if calculate_sha256:
                    h_sha256.update(chunk)
                if calculate_md5:
                    h_md5.update(chunk)
        if calculate_sha256:
            assets[ii]['checksum_sha256'] = h_sha256.hexdigest()
        if calculate_md5:
            assets[ii]['checksum_md5'] = h_md5.hexdigest()

    with Image.open(image_path) as image:
        if has_transparency(image):
            image = image.convert('RGBA')
            background = Image.new('RGBA', image.size, (background_intensity,) * 3)
            composite = Image.alpha_composite(background, image)
            image = composite.convert('RGB')
        else:
            image = image.convert('RGB')

        if image.size[0] != image.size[1]:
            # print(f'{image_file} is not square, padding...')
            image = ImageOps.pad(image, size=(np.max(image.size[:2]),) * 2, color=(background_intensity,) * 3)

    #  * * * * * CONSIDER MULTIPLE IMAGE BACKGROUNDS AS WELL

    if 'alexnet' not in assets[ii]:
        assets[ii]['alexnet'] = {
            '': ,
        }

    # based on Bao et al Tsao 2020 Nature
    #   "Decoding accuracy for 40 images using object spaces built by responses of different layers of AlexNet
    #   (computed as in Extended Data Fig. 11d). There are multiple points for each layer because we performed
    #   PCA before and after pooling, activation, and normalization functions. Layer fc6 showed the highest
    #   decoding accuracy, motivating our use of the object space generated by this layer throughout the paper."

    # AlexNet
    # features conv1
    alexnet_00_conv1_conv2d = alexnet.features[0]
    alexnet_00_conv1_conv2d.register_forward_hook(get_activation('alexnet_00_conv1_conv2d'))
    alexnet_01_conv1_relu = alexnet.features[1]
    alexnet_01_conv1_relu.register_forward_hook(get_activation('alexnet_01_conv1_relu'))
    alexnet_02_conv1_maxpool2d = alexnet.features[2]
    alexnet_02_conv1_maxpool2d.register_forward_hook(get_activation('alexnet_02_conv1_maxpool2d'))
    # features conv2
    alexnet_03_conv2_conv2d = alexnet.features[3]
    alexnet_03_conv2_conv2d.register_forward_hook(get_activation('alexnet_03_conv2_conv2d'))
    alexnet_04_conv2_relu = alexnet.features[4]
    alexnet_04_conv2_relu.register_forward_hook(get_activation('alexnet_04_conv2_relu'))
    alexnet_05_conv2_maxpool2d = alexnet.features[5]
    alexnet_05_conv2_maxpool2d.register_forward_hook(get_activation('alexnet_05_conv2_maxpool2d'))
    # features conv3
    alexnet_06_conv3_conv2d = alexnet.features[6]
    alexnet_06_conv3_conv2d.register_forward_hook(get_activation('alexnet_06_conv3_conv2d'))
    alexnet_07_conv3_relu = alexnet.features[7]
    alexnet_07_conv3_relu.register_forward_hook(get_activation('alexnet_07_conv3_relu'))
    # features conv4
    alexnet_08_conv4_conv2d = alexnet.features[8]
    alexnet_08_conv4_conv2d.register_forward_hook(get_activation('alexnet_08_conv4_conv2d'))
    alexnet_09_conv4_relu = alexnet.features[9]
    alexnet_09_conv4_relu.register_forward_hook(get_activation('alexnet_09_conv4_relu'))
    # features conv5
    alexnet_10_conv5_conv2d = alexnet.features[10]
    alexnet_10_conv5_conv2d.register_forward_hook(get_activation('alexnet_10_conv5_conv2d'))
    alexnet_11_conv5_relu = alexnet.features[11]
    alexnet_11_conv5_relu.register_forward_hook(get_activation('alexnet_11_conv5_relu'))
    alexnet_12_conv5_maxpool2d = alexnet.features[12]
    alexnet_12_conv5_maxpool2d.register_forward_hook(get_activation('alexnet_12_conv5_maxpool2d'))

    # avgpool
    alexnet_13_avgpool = alexnet.avgpool
    alexnet_13_avgpool.register_forward_hook(get_activation('alexnet_13_avgpool'))

    # classifier fc6
    alexnet_14_fc6_dropout = alexnet.classifier[0]
    alexnet_14_fc6_dropout.register_forward_hook(get_activation('alexnet_14_fc6_dropout'))
    alexnet_15_fc6_linear = alexnet.classifier[1]
    alexnet_15_fc6_linear.register_forward_hook(get_activation('alexnet_15_fc6_linear'))
    alexnet_16_fc6_relu = alexnet.classifier[2]
    alexnet_16_fc6_relu.register_forward_hook(get_activation('alexnet_16_fc6_relu'))

    # classifier fc7
    alexnet_17_fc7_dropout = alexnet.classifier[3]
    alexnet_17_fc7_dropout.register_forward_hook(get_activation('alexnet_17_fc7_dropout'))
    alexnet_18_fc7_linear = alexnet.classifier[4]
    alexnet_18_fc7_linear.register_forward_hook(get_activation('alexnet_18_fc7_linear'))
    alexnet_19_fc7_relu = alexnet.classifier[5]
    alexnet_19_fc7_relu.register_forward_hook(get_activation('alexnet_19_fc7_relu'))

    # classifier fc8
    alexnet_20_fc8_linear = alexnet.classifier[6]
    alexnet_20_fc8_linear.register_forward_hook(get_activation('alexnet_20_fc8_linear'))

    # fc6 = alexnet.classifier[1]  # output from 'fc6' (1): Linear(in_features=9216, out_features=4096, bias=True)
    # fc6.register_forward_hook(get_activation('fc6'))
    # fc6relu = alexnet.classifier[2]
    # fc6relu.register_forward_hook(get_activation('fc6relu'))
    # fc7 = alexnet.classifier[4]  # output from 'fc6' (1): Linear(in_features=9216, out_features=4096, bias=True)
    # fc7.register_forward_hook(get_activation('fc7'))
    # fc7relu = alexnet.classifier[5]
    # fc7relu.register_forward_hook(get_activation('fc7relu'))
    #
    # fc6_features = np.full([n_images, 4096], np.nan)
    # fc6relu_features = np.full([n_images, 4096], np.nan)
    # fc7_features = np.full([n_images, 4096], np.nan)
    # fc7relu_features = np.full([n_images, 4096], np.nan)

    preprocess = transforms.Compose([
        transforms.Resize(224),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = preprocess(image)
    input_batch = tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device)
    alexnet.to(device)

    with torch.no_grad():
        output = alexnet(input_batch)

    fc6_features[i_im, :] = activation['fc6'].cpu().numpy().squeeze()
    # fc6relu_features[i_im, :] = activation['fc6relu'].cpu().numpy().squeeze()
    fc7_features[i_im, :] = activation['fc7'].cpu().numpy().squeeze()
    # fc7relu_features[i_im, :] = activation['fc7relu'].cpu().numpy().squeeze()

    # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    print(image_file, categories[top_catid], top_prob.item())


# from Bao et al Tsao 2020 Nature
# " Decoding accuracy for 40 images using object spaces built by responses of different layers of AlexNet
#  ... There are multiple points for each layer because we performed PCA before and after pooling, activation,
#  and normalization functions. Layer fc6 showed the highest decoding accuracy, motivating our use of the object
#  space generated by this layer throughout the paper."





# def calculate_sha256_hash(file_path):
#     hash_sha256 = hashlib.sha256()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             hash_sha256.update(chunk)
#     return hash_sha256.hexdigest()
#
#
# def calculate_md5_hash(file_path):
#     hash_md5 = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             hash_md5.update(chunk)
#     return hash_md5.hexdigest()
#
#
# def checksum_file(file_path):
#     h_sha256 = hashlib.sha256()
#     h_md5 = hashlib.md5()
#     with open(file_path, 'rb') as f:
#         for chunk in iter(lambda: f.read(4096), b''):
#             h_sha256.update(chunk)
#             h_md5.update(chunk)
#     return h_sha256.hexdigest(), h_md5.hexdigest()
