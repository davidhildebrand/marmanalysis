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


image_path = r'TsaoStimuli\500Stimuli'

# min_dimension = 299
min_dimension = 224
pad_color = 255

checksumming = True
skipping = False

skiplist = []


def has_transparency(img):
    # based on https://stackoverflow.com/questions/43864101/python-pil-check-if-image-is-transparent
    if img.mode == 'LA':
        img = image.convert('RGBA')
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
            '00_conv1_conv2d': activ['features.0'].cpu().numpy().squeeze(),
            '01_conv1_relu': activ['features.1'].cpu().numpy().squeeze(),
            '02_conv1_maxpool2d': activ['features.2'].cpu().numpy().squeeze(),
            '03_conv2_conv2d': activ['features.3'].cpu().numpy().squeeze(),
            '04_conv2_relu': activ['features.4'].cpu().numpy().squeeze(),
            '05_conv2_maxpool2d': activ['features.5'].cpu().numpy().squeeze(),
            '06_conv3_conv2d': activ['features.6'].cpu().numpy().squeeze(),
            '07_conv3_relu': activ['features.7'].cpu().numpy().squeeze(),
            '08_conv4_conv2d': activ['features.8'].cpu().numpy().squeeze(),
            '09_conv4_relu': activ['features.9'].cpu().numpy().squeeze(),
            '10_conv5_conv2d': activ['features.10'].cpu().numpy().squeeze(),
            '11_conv5_relu': activ['features.11'].cpu().numpy().squeeze(),
            '12_conv5_maxpool2d': activ['features.12'].cpu().numpy().squeeze(),
            '13_avgpool': activ['avgpool'].cpu().numpy().squeeze(),
            '14_fc6_dropout': activ['classifier.0'].cpu().numpy().squeeze(),
            '15_fc6_linear': activ['classifier.1'].cpu().numpy().squeeze(),
            '16_fc6_relu': activ['classifier.2'].cpu().numpy().squeeze(),
            '17_fc7_dropout': activ['classifier.3'].cpu().numpy().squeeze(),
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


def save_snapshot(savestr='_asset_unit_responses'):
    now = datetime.now(timezone.utc)
    datetime_str = now.strftime('%Y%m%dd%H%M%StUTC')
    infosave_filename = datetime_str + savestr + '.pickle'
    infosave_filepath = os.path.join(infosave_path, infosave_filename)
    with open(infosave_filepath, 'wb') as pickle_file:
        pickle.dump([assets], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


system_name = socket.gethostname()
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/ImageDatasets'
    device = 'cpu'
    # device = 'mps' if torch.backends.mps.is_available() else 'cpu'
elif 'marmostor' in system_name:
    base_path = r'/marmostor/DavidH/ImageDatasets'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data\ImageDatasets'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data\ImageDatasets'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
else:
    base_path = None
    asset_path = None
    tree_path = None
    infosave_path = None
    device = 'cpu'
    warn('Unknown system name, paths not set.')

asset_path = base_path + os.path.sep + image_path
# tree_path = base_path + os.path.sep + 'category_tree'
infosave_path = base_path + os.path.sep + 'info_saves'


# Model implementation from...
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


# from Bao et al Tsao 2020 Nature
# " Decoding accuracy for 40 images using object spaces built by responses of different layers of AlexNet
#  ... There are multiple points for each layer because we performed PCA before and after pooling, activation,
#  and normalization functions. Layer fc6 showed the highest decoding accuracy, motivating our use of the object
#  space generated by this layer throughout the paper."
# based on Bao et al Tsao 2020 Nature
#   "Decoding accuracy for 40 images using object spaces built by responses of different layers of AlexNet
#   (computed as in Extended Data Fig. 11d). There are multiple points for each layer because we performed
#   PCA before and after pooling, activation, and normalization functions. Layer fc6 showed the highest
#   decoding accuracy, motivating our use of the object space generated by this layer throughout the paper."


image_files = [f for f in os.listdir(asset_path) if filetype.is_image(os.path.join(asset_path, f))]

assets = {}
asset_counter = 0

for i in image_files:
    if skipping:
        if i in skiplist:
            print('Skipped {}... skiplisted.'.format(i))
            continue

    if i not in assets:
        assets[i] = {
            'filename': i,
            'filepath': os.path.join(asset_path, i),
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
        calculate_sha256 = False
        calculate_md5 = False
        if 'checksum_sha256' not in assets[i]:
            calculate_sha256 = True
        if 'checksum_md5' not in assets[i]:
            calculate_md5 = True
        if calculate_sha256 or calculate_md5:
            h_sha256 = hashlib.sha256()
            h_md5 = hashlib.md5()
            with open(assets[i]['filepath'], 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    if calculate_sha256:
                        h_sha256.update(chunk)
                    if calculate_md5:
                        h_md5.update(chunk)
            if calculate_sha256:
                assets[i]['checksum_sha256'] = h_sha256.hexdigest()
            if calculate_md5:
                assets[i]['checksum_md5'] = h_md5.hexdigest()
        del h_sha256, h_md5

    if 'image_size' in assets[i]:
        if np.min(assets[i]['image_size']) < min_dimension:
            print('Skipped {}... {} is less than minimum side {} px.'.format(i,
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

            for k in assets[i]['unit_responses']['alexnet'].keys():
                print('{}: shape = {}, mean = {}'.format(k,
                                                         assets[i]['unit_responses']['alexnet'][k].shape,
                                                         np.mean(assets[i]['unit_responses']['alexnet'][k])))
            # del activation, tensor, input_batch, model_output

            # probabilities = torch.nn.functional.softmax(model_output[0], dim=0)
            # print(probabilities)
            # with open(r'F:\Data\ImageDatasets\imagenet_classes.txt', 'r') as f:
            #     categories = [s.strip() for s in f.readlines()]
            # # Show top categories per image
            # top5_prob, top5_catid = torch.topk(probabilities, 5)
            # for ii in range(top5_prob.size(0)):
            #     print(categories[top5_catid[ii]], top5_prob[ii].item())

    if asset_counter % 100 == 0 and asset_counter > 0:
        save_snapshot(savestr='_asset_unit_responses_alexnet_partial')
    asset_counter += 1
save_snapshot(savestr='_asset_unit_responses_full')


# # AlexNet features conv1
# alexnet_hook00_conv1_conv2d = alexnet.features[0]
# h1 = alexnet_hook00_conv1_conv2d.register_forward_hook(get_activation('alexnet_hook00_conv1_conv2d'))
# alexnet_hook01_conv1_relu = alexnet.features[1]
# alexnet_hook01_conv1_relu.register_forward_hook(get_activation('alexnet_hook01_conv1_relu'))
# alexnet_hook02_conv1_maxpool2d = alexnet.features[2]
# alexnet_hook02_conv1_maxpool2d.register_forward_hook(get_activation('alexnet_hook02_conv1_maxpool2d'))
# # AlexNet features conv2
# alexnet_hook03_conv2_conv2d = alexnet.features[3]
# alexnet_hook03_conv2_conv2d.register_forward_hook(get_activation('alexnet_hook03_conv2_conv2d'))
# alexnet_hook04_conv2_relu = alexnet.features[4]
# alexnet_hook04_conv2_relu.register_forward_hook(get_activation('alexnet_hook04_conv2_relu'))
# alexnet_hook05_conv2_maxpool2d = alexnet.features[5]
# alexnet_hook05_conv2_maxpool2d.register_forward_hook(get_activation('alexnet_hook05_conv2_maxpool2d'))
# # AlexNet features conv3
# alexnet_hook06_conv3_conv2d = alexnet.features[6]
# alexnet_hook06_conv3_conv2d.register_forward_hook(get_activation('alexnet_hook06_conv3_conv2d'))
# alexnet_hook07_conv3_relu = alexnet.features[7]
# alexnet_hook07_conv3_relu.register_forward_hook(get_activation('alexnet_hook07_conv3_relu'))
# # AlexNet features conv4
# alexnet_hook08_conv4_conv2d = alexnet.features[8]
# alexnet_hook08_conv4_conv2d.register_forward_hook(get_activation('alexnet_hook08_conv4_conv2d'))
# alexnet_hook09_conv4_relu = alexnet.features[9]
# alexnet_hook09_conv4_relu.register_forward_hook(get_activation('alexnet_hook09_conv4_relu'))
# # AlexNet features conv5
# alexnet_hook10_conv5_conv2d = alexnet.features[10]
# alexnet_hook10_conv5_conv2d.register_forward_hook(get_activation('alexnet_hook10_conv5_conv2d'))
# alexnet_hook11_conv5_relu = alexnet.features[11]
# alexnet_hook11_conv5_relu.register_forward_hook(get_activation('alexnet_hook11_conv5_relu'))
# alexnet_hook12_conv5_maxpool2d = alexnet.features[12]
# alexnet_hook12_conv5_maxpool2d.register_forward_hook(get_activation('alexnet_hook12_conv5_maxpool2d'))
# # AlexNet avgpool
# alexnet_hook13_avgpool = alexnet.avgpool
# alexnet_hook13_avgpool.register_forward_hook(get_activation('alexnet_hook13_avgpool'))
# # AlexNet classifier fc6
# alexnet_hook14_fc6_dropout = alexnet.classifier[0]
# alexnet_hook14_fc6_dropout.register_forward_hook(get_activation('alexnet_hook14_fc6_dropout'))
# alexnet_hook15_fc6_linear = alexnet.classifier[1]
# alexnet_hook15_fc6_linear.register_forward_hook(get_activation('alexnet_hook15_fc6_linear'))
# alexnet_hook16_fc6_relu = alexnet.classifier[2]
# alexnet_hook16_fc6_relu.register_forward_hook(get_activation('alexnet_hook16_fc6_relu'))
# # AlexNet classifier fc7
# alexnet_hook17_fc7_dropout = alexnet.classifier[3]
# alexnet_hook17_fc7_dropout.register_forward_hook(get_activation('alexnet_hook17_fc7_dropout'))
# alexnet_hook18_fc7_linear = alexnet.classifier[4]
# alexnet_hook18_fc7_linear.register_forward_hook(get_activation('alexnet_hook18_fc7_linear'))
# alexnet_hook19_fc7_relu = alexnet.classifier[5]
# alexnet_hook19_fc7_relu.register_forward_hook(get_activation('alexnet_hook19_fc7_relu'))
# # AlexNet classifier fc8
# alexnet_hook20_fc8_linear = alexnet.classifier[6]
# alexnet_hook20_fc8_linear.register_forward_hook(get_activation('alexnet_hook20_fc8_linear'))
