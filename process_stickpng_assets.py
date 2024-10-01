#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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


image_info_file = r'20240908d203128tUTC_stickpng_image_info_full.pickle'
min_dimension = 299

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
def has_transparency(img):
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


def generate_pinknoise_image(size=(512, 512), intensity=128):
    # Ported from MATLAB function 'pinkinoiseimage' used by Xindong Song
    # https://github.com/x-song-x/fluffy-goggles/blob/master/XinStimEx/XinStimEx_Vis_FacePatch_Trail.m
    beta = 1
    fhh = np.ceil((size[0] / 2) + 1).astype(int)
    fhw = np.ceil((size[1] / 2) + 1).astype(int)
    fhimagea = np.arange(0, fhh)[:,np.newaxis] @ np.ones([1, fhw])
    fhimageb = np.ones([fhh, 1]) @ np.arange(0, fhw)[np.newaxis,:]
    fhimagef = (fhimagea ** 2 + fhimageb ** 2) ** (1/2)
    np.seterr(divide='ignore')
    fhimagefbeta = fhimagef ** (-beta)
    np.seterr(divide='warn')
    fhimagefbeta[0,0] = 0
    fhimageagl1 = np.random.rand(fhh, fhw) * 2 * np.pi
    fhimageagl1[-1,0] = 0
    fhimageagl1[0,-1] = 0
    fhimageagl1[-1,-1] = 0
    fhimageagl2 = np.random.rand(fhh, fhw) * 2 * np.pi
    fhimagecomp1 = np.empty(fhimageagl1.shape, dtype=complex)
    fhimagecomp1.real = fhimagefbeta * np.cos(fhimageagl1)
    fhimagecomp1.imag = fhimagefbeta * np.sin(fhimageagl1)
    fhimagecomp2 = np.empty(fhimageagl2.shape, dtype=complex)
    fhimagecomp2.real = fhimagefbeta * np.cos(fhimageagl2)
    fhimagecomp2.imag = fhimagefbeta * np.sin(fhimageagl2)
    fimagecomp = np.concatenate((fhimagecomp1,
                                np.fliplr(np.concatenate((np.conj(fhimagecomp1[0,1:-1][np.newaxis,:]),
                                                          fhimagecomp2[1:-1,1:-1],
                                                          np.conj(fhimagecomp1[-1,1:-1][np.newaxis,:]))))), 1)
    fimagecomp = np.concatenate((fimagecomp,
                                 np.conj(np.flipud(np.concatenate((fhimagecomp1[1:-1,0][:,np.newaxis],
                                                                  np.fliplr(fimagecomp[1:-1,1::])), 1)))))
    imagetmp = np.fft.ifft2(fimagecomp)
    imagemax = np.max(imagetmp)
    imagemin = np.min(imagetmp)
    imagemaxn = np.abs(imagemax / (255 - intensity))
    imageminn = np.abs(imagemin / (intensity - 0))
    imageampn = np.max([imagemaxn, imageminn])
    img_pink = (imagetmp.real / imageampn) + intensity
    # img_pink = skimage.exposure.rescale_intensity(img_pink, in_range=(0, 255), out_range=(-1, 1))
    if img_pink.size != size:
        img_pink = img_pink[:size[0], :size[1]]
    return img_pink


# Create hook for extracting intermediate layer output
# based on https://discuss.pytorch.org/t/how-can-i-extract-intermediate-layer-output-from-loaded-cnn-model/77301/3
#     and https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/14
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def get_modvals_alexnet():
    if 'activation' in locals():
        d = {
            '00_conv1_conv2d': activation['features.0'].cpu().numpy().squeeze(),
            '01_conv1_relu': activation['features.1'].cpu().numpy().squeeze(),
            '02_conv1_maxpool2d': activation['features.2'].cpu().numpy().squeeze(),
            '03_conv2_conv2d': activation['features.3'].cpu().numpy().squeeze(),
            '04_conv2_relu': activation['features.4'].cpu().numpy().squeeze(),
            '05_conv2_maxpool2d': activation['features.5'].cpu().numpy().squeeze(),
            '06_conv3_conv2d': activation['features.6'].cpu().numpy().squeeze(),
            '07_conv3_relu': activation['features.7'].cpu().numpy().squeeze(),
            '08_conv4_conv2d': activation['features.8'].cpu().numpy().squeeze(),
            '09_conv4_relu': activation['features.9'].cpu().numpy().squeeze(),
            '10_conv5_conv2d': activation['features.10'].cpu().numpy().squeeze(),
            '11_conv5_relu': activation['features.11'].cpu().numpy().squeeze(),
            '12_conv5_maxpool2d': activation['features.12'].cpu().numpy().squeeze(),
            '13_avgpool': activation['avgpool'].cpu().numpy().squeeze(),
            '14_fc6_dropout': activation['classifier.0'].cpu().numpy().squeeze(),
            '15_fc6_linear': activation['classifier.1'].cpu().numpy().squeeze(),
            '16_fc6_relu': activation['classifier.2'].cpu().numpy().squeeze(),
            '17_fc7_dropout': activation['classifier.3'].cpu().numpy().squeeze(),
            '18_fc7_linear': activation['classifier.4'].cpu().numpy().squeeze(),
            '19_fc7_relu': activation['classifier.5'].cpu().numpy().squeeze(),
            '20_fc8_linear': activation['classifier.6'].cpu().numpy().squeeze(),
        }
    else:
        warn('No activations found.')
        d = {}
    return d


def save_snapshot(savestr='_asset_modvals'):
    now = datetime.now(timezone.utc)
    datetime_str = now.strftime('%Y%m%dd%H%M%StUTC')
    infosave_filename = datetime_str + savestr + '.pickle'
    infosave_filepath = os.path.join(infosave_path, infosave_filename)
    with open(infosave_filepath, 'wb') as pickle_file:
        pickle.dump([assets], pickle_file, protocol=pickle.HIGHEST_PROTOCOL)


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
    asset_path = base_path + os.path.sep + 'assets'
    tree_path = base_path + os.path.sep + 'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Obsidian' in system_name:
    base_path = r'F:\Data\ImageDatasets\stickpng'
    asset_path = base_path + os.path.sep + 'assets'
    tree_path = base_path + os.path.sep + 'category_tree'
    infosave_path = base_path + os.path.sep + 'info_saves'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
elif 'Dobbin' in system_name:
    base_path = r'D:\Data\ImageDatasets\stickpng'
    asset_path = base_path + os.path.sep + 'assets'
    tree_path = base_path + os.path.sep + 'category_tree'
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


# # How background pink noise images were created (now loaded to keep background consistent)
# background_224_pinkn = Image.fromarray(generate_pinknoise_image(size=(224, 224), intensity=128).T.astype(int))
# background_224_pinkn = background_224_pinkn.convert('RGB')
# background_224_pinkn.save(os.path.join(base_path, 'backgrounds', 'background_224_pinknoise.png'))
#
# background_299_pinkn = Image.fromarray(generate_pinknoise_image(size=(299, 299), intensity=128).T.astype(int))
# background_299_pinkn = background_299_pinkn.convert('RGB')
# background_299_pinkn.save(os.path.join(base_path, 'backgrounds', 'background_299_pinknoise.png'))
#
# background_512_pinkn = Image.fromarray(generate_pinknoise_image(size=(512, 512), intensity=128).T.astype(int))
# background_512_pinkn = background_512_pinkn.convert('RGB')
# background_512_pinkn.save(os.path.join(base_path, 'backgrounds', 'background_512_pinknoise.png'))


# Model implementations from...
# alexnet: https://pytorch.org/hub/pytorch_vision_alexnet/
# vgg-nets: https://pytorch.org/hub/pytorch_vision_vgg/
# googlenet: https://pytorch.org/hub/pytorch_vision_googlenet/
# inceptionv3: https://pytorch.org/hub/pytorch_vision_inception_v3/
# resnet: https://pytorch.org/hub/pytorch_vision_resnet/
# resnext: https://pytorch.org/hub/pytorch_vision_resnext/
# wide_resnet: https://pytorch.org/hub/pytorch_vision_wide_resnet/
# densenet: https://pytorch.org/hub/pytorch_vision_densenet/

hooks = {}

alexnet = torch.hub.load('pytorch/vision:v0.19.1', 'alexnet', weights='AlexNet_Weights.IMAGENET1K_V1')
hooks['alexnet'] = {}
for name, module in alexnet.named_modules():
    hooks['alexnet'][name] = module.register_forward_hook(get_activation(name))

vgg11 = torch.hub.load('pytorch/vision:v0.19.1', 'vgg11', weights='VGG11_Weights.IMAGENET1K_V1')
hooks['vgg11'] = {}
for name, module in vgg11.named_modules():
    hooks['vgg11'][name] = module.register_forward_hook(get_activation(name))

vgg11bn = torch.hub.load('pytorch/vision:v0.19.1', 'vgg11_bn', weights='VGG11_BN_Weights.IMAGENET1K_V1')
hooks['vgg11bn'] = {}
for name, module in vgg11bn.named_modules():
    hooks['vgg11bn'][name] = module.register_forward_hook(get_activation(name))

vgg13 = torch.hub.load('pytorch/vision:v0.19.1', 'vgg13', weights='VGG13_Weights.IMAGENET1K_V1')
hooks['vgg13'] = {}
for name, module in vgg13.named_modules():
    hooks['vgg13'][name] = module.register_forward_hook(get_activation(name))

vgg13bn = torch.hub.load('pytorch/vision:v0.19.1', 'vgg13_bn', weights='VGG13_BN_Weights.IMAGENET1K_V1')
hooks['vgg13bn'] = {}
for name, module in vgg13bn.named_modules():
    hooks['vgg13bn'][name] = module.register_forward_hook(get_activation(name))

vgg16 = torch.hub.load('pytorch/vision:v0.19.1', 'vgg16', weights='VGG16_Weights.IMAGENET1K_V1')
hooks['vgg16'] = {}
for name, module in vgg16.named_modules():
    hooks['vgg16'][name] = module.register_forward_hook(get_activation(name))

vgg16bn = torch.hub.load('pytorch/vision:v0.19.1', 'vgg16_bn', weights='VGG16_BN_Weights.IMAGENET1K_V1')
hooks['vgg16bn'] = {}
for name, module in vgg16bn.named_modules():
    hooks['vgg16bn'][name] = module.register_forward_hook(get_activation(name))

vgg19 = torch.hub.load('pytorch/vision:v0.19.1', 'vgg19', weights='VGG19_Weights.IMAGENET1K_V1')
hooks['vgg19'] = {}
for name, module in vgg19.named_modules():
    hooks['vgg19'][name] = module.register_forward_hook(get_activation(name))

vgg19bn = torch.hub.load('pytorch/vision:v0.19.1', 'vgg19_bn', weights='VGG19_BN_Weights.IMAGENET1K_V1')
hooks['vgg19bn'] = {}
for name, module in vgg19bn.named_modules():
    hooks['vgg19bn'][name] = module.register_forward_hook(get_activation(name))

googlenet = torch.hub.load('pytorch/vision:v0.19.1', 'googlenet', weights='GoogLeNet_Weights.IMAGENET1K_V1')
hooks['googlenet'] = {}
for name, module in googlenet.named_modules():
    hooks['googlenet'][name] = module.register_forward_hook(get_activation(name))

inceptionv3 = torch.hub.load('pytorch/vision:v0.19.1', 'inception_v3', weights='Inception_V3_Weights.IMAGENET1K_V1')
hooks['inceptionv3'] = {}
for name, module in inceptionv3.named_modules():
    hooks['inceptionv3'][name] = module.register_forward_hook(get_activation(name))

resnet18 = torch.hub.load('pytorch/vision:v0.19.1', 'resnet18', weights='ResNet18_Weights.IMAGENET1K_V1')
hooks['resnet18'] = {}
for name, module in resnet18.named_modules():
    hooks['resnet18'][name] = module.register_forward_hook(get_activation(name))

resnet34 = torch.hub.load('pytorch/vision:v0.19.1', 'resnet34', weights='ResNet34_Weights.IMAGENET1K_V1')
hooks['resnet34'] = {}
for name, module in resnet34.named_modules():
    hooks['resnet34'][name] = module.register_forward_hook(get_activation(name))

resnet50 = torch.hub.load('pytorch/vision:v0.19.1', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V2')
hooks['resnet50'] = {}
for name, module in resnet50.named_modules():
    hooks['resnet50'][name] = module.register_forward_hook(get_activation(name))

resnet101 = torch.hub.load('pytorch/vision:v0.19.1', 'resnet101', weights='ResNet101_Weights.IMAGENET1K_V2')
hooks['resnet101'] = {}
for name, module in resnet101.named_modules():
    hooks['resnet101'][name] = module.register_forward_hook(get_activation(name))

resnet152 = torch.hub.load('pytorch/vision:v0.19.1', 'resnet152', weights='ResNet152_Weights.IMAGENET1K_V2')
hooks['resnet152'] = {}
for name, module in resnet152.named_modules():
    hooks['resnet152'][name] = module.register_forward_hook(get_activation(name))

resnext5032x4d = torch.hub.load('pytorch/vision:v0.19.1', 'resnext50_32x4d', weights='ResNeXt50_32X4D_Weights.IMAGENET1K_V2')
hooks['resnext5032x4d'] = {}
for name, module in resnext5032x4d.named_modules():
    hooks['resnext5032x4d'][name] = module.register_forward_hook(get_activation(name))

resnext10132x8d = torch.hub.load('pytorch/vision:v0.19.1', 'resnext101_32x8d', weights='ResNeXt101_32X8D_Weights.IMAGENET1K_V2')
hooks['resnext10132x8d'] = {}
for name, module in resnext10132x8d.named_modules():
    hooks['resnext10132x8d'][name] = module.register_forward_hook(get_activation(name))

resnext10164x4d = torch.hub.load('pytorch/vision:v0.19.1', 'resnext101_64x4d', weights='ResNeXt101_64X4D_Weights.IMAGENET1K_V1')
hooks['resnext10164x4d'] = {}
for name, module in resnext10164x4d.named_modules():
    hooks['resnext10164x4d'][name] = module.register_forward_hook(get_activation(name))

wideresnet502 = torch.hub.load('pytorch/vision:v0.19.1', 'wide_resnet50_2', weights='Wide_ResNet50_2_Weights.IMAGENET1K_V2')
hooks['wideresnet502'] = {}
for name, module in wideresnet502.named_modules():
    hooks['wideresnet502'][name] = module.register_forward_hook(get_activation(name))

wideresnet1012 = torch.hub.load('pytorch/vision:v0.19.1', 'wide_resnet101_2', weights='Wide_ResNet101_2_Weights.IMAGENET1K_V2')
hooks['wideresnet1012'] = {}
for name, module in wideresnet1012.named_modules():
    hooks['wideresnet1012'][name] = module.register_forward_hook(get_activation(name))

densenet121 = torch.hub.load('pytorch/vision:v0.19.1', 'densenet121', weights='DenseNet121_Weights.IMAGENET1K_V1')
hooks['densenet121'] = {}
for name, module in densenet121.named_modules():
    hooks['densenet121'][name] = module.register_forward_hook(get_activation(name))

densenet161 = torch.hub.load('pytorch/vision:v0.19.1', 'densenet161', weights='DenseNet161_Weights.IMAGENET1K_V1')
hooks['densenet161'] = {}
for name, module in densenet161.named_modules():
    hooks['densenet161'][name] = module.register_forward_hook(get_activation(name))

densenet169 = torch.hub.load('pytorch/vision:v0.19.1', 'densenet169', weights='DenseNet169_Weights.IMAGENET1K_V1')
hooks['densenet169'] = {}
for name, module in densenet169.named_modules():
    hooks['densenet169'][name] = module.register_forward_hook(get_activation(name))

densenet201 = torch.hub.load('pytorch/vision:v0.19.1', 'densenet201', weights='DenseNet201_Weights.IMAGENET1K_V1')
hooks['densenet201'] = {}
for name, module in densenet201.named_modules():
    hooks['densenet201'][name] = module.register_forward_hook(get_activation(name))

squeezenet10 = torch.hub.load('pytorch/vision:v0.19.1', 'squeezenet1_0', weights='SqueezeNet1_0_Weights.IMAGENET1K_V1')
hooks['squeezenet10'] = {}
for name, module in squeezenet10.named_modules():
    hooks['squeezenet10'][name] = module.register_forward_hook(get_activation(name))

squeezenet11 = torch.hub.load('pytorch/vision:v0.19.1', 'squeezenet1_1', weights='SqueezeNet1_1_Weights.IMAGENET1K_V1')
hooks['squeezenet11'] = {}
for name, module in squeezenet11.named_modules():
    hooks['squeezenet11'][name] = module.register_forward_hook(get_activation(name))

# Load pinknoise backgrounds into memory
background_224_pinkn = Image.open(os.path.join(base_path, 'backgrounds', 'background_224_pinknoise.png'))
background_224_pinkn = background_224_pinkn.convert('RGBA')
background_299_pinkn = Image.open(os.path.join(base_path, 'backgrounds', 'background_299_pinknoise.png'))
background_299_pinkn = background_299_pinkn.convert('RGBA')


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


assets = {}
asset_counter = 0
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
            'categorysub': image_info[ii]['code_subcategory'],
            'categoryfull': image_info[ii]['code_category_full'],
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

    if 'image_size' in assets[ii]:
        if np.min(assets[ii]['image_size']) < min_dimension:
            warn('Skipped {} because it is smaller than {} px in at least one dimension.'.format(ii, min_dimension))
            continue

    # ...at least according to PyTorch documentation, the normalization values are the same for all these models.
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with Image.open(image_path) as image:
        assets[ii]['image_size'] = image.size
        if np.min(assets[ii]['image_size']) < min_dimension:
            warn('Skipped {} because it is smaller than {} px in at least one dimension.'.format(ii, min_dimension))
            continue
        if has_transparency(image):
            assets[ii]['image_transparency'] = True
            image = image.convert('RGBA')

            resize_ratio_224 = np.min((224 / image.size[0], 224 / image.size[1]))
            image_224 = image.resize(np.round(resize_ratio_224 * np.array(image.size)).astype(int),
                                     resample=Image.Resampling.BICUBIC)
            if image_224.size[0] != image_224.size[1]:
                image_224 = ImageOps.pad(image_224, size=(np.max(image_224.size[:2]),) * 2, color=(255, 255, 255, 0))
            background_224_white = Image.new('RGBA', image_224.size, (255,) * 3)
            background_224_gray = Image.new('RGBA', image_224.size, (128,) * 3)
            # background_224_pinkn = Image.fromarray(generate_pinknoise_image(size=image_224.size, intensity=128).T.astype(int))
            # background_224_pinkn = background_224_pinkn.convert('RGBA')
            # with Image.open(os.path.join(base_path, 'backgrounds', 'background_224_pinknoise.png')) as pinkn_224:
            #     background_224_pinkn = pinkn_224
            # background_224_pinkn = background_224_pinkn.convert('RGBA')
            composite_224_white = Image.alpha_composite(background_224_white, image_224)
            composite_224_gray = Image.alpha_composite(background_224_gray, image_224)
            composite_224_pinkn = Image.alpha_composite(background_224_pinkn, image_224)
            image_224_white = composite_224_white.convert('RGB')
            image_224_gray = composite_224_gray.convert('RGB')
            image_224_pinkn = composite_224_pinkn.convert('RGB')

            resize_ratio_299 = np.min((299 / image.size[0], 299 / image.size[1]))
            image_299 = image.resize(np.round(resize_ratio_299 * np.array(image.size)).astype(int),
                                     resample=Image.Resampling.BICUBIC)
            if image_299.size[0] != image_299.size[1]:
                image_299 = ImageOps.pad(image_299, size=(np.max(image_299.size[:2]),) * 2, color=(255, 255, 255, 0))
            background_299_white = Image.new('RGBA', image_299.size, (255,) * 3)
            background_299_gray = Image.new('RGBA', image_299.size, (128,) * 3)
            # background_299_pinkn = Image.fromarray(generate_pinknoise_image(size=image_299.size, intensity=128).T.astype(int))
            # background_299_pinkn = background_299_pinkn.convert('RGBA')
            # with Image.open(os.path.join(base_path, 'backgrounds', 'background_299_pinknoise.png')) as pinkn_299:
            #     background_299_pinkn = pinkn_299
            # background_299_pinkn = background_299_pinkn.convert('RGBA')
            composite_299_white = Image.alpha_composite(background_299_white, image_299)
            composite_299_gray = Image.alpha_composite(background_299_gray, image_299)
            composite_299_pinkn = Image.alpha_composite(background_299_pinkn, image_299)
            image_299_white = composite_299_white.convert('RGB')
            image_299_gray = composite_299_gray.convert('RGB')
            image_299_pinkn = composite_299_pinkn.convert('RGB')

            if 'modvals' not in assets[ii]:
                assets[ii]['modvals'] = {}

            if 'alexnet' not in assets[ii]['modvals']:
                assets[ii]['modvals']['alexnet'] = {}

                activation = {}
                tensor = preprocess(image_224_white)
                input_batch = tensor.unsqueeze(0)
                input_batch = input_batch.to(device)
                alexnet.to(device)
                with torch.no_grad():
                    output = alexnet(input_batch)
                # assets[ii]['modvals']['alexnet']['bg_w'] = {
                #     '00_conv1_conv2d': activation['features.0'].cpu().numpy().squeeze(),
                #     '01_conv1_relu': activation['features.1'].cpu().numpy().squeeze(),
                #     '02_conv1_maxpool2d': activation['features.2'].cpu().numpy().squeeze(),
                #     '03_conv2_conv2d': activation['features.3'].cpu().numpy().squeeze(),
                #     '04_conv2_relu': activation['features.4'].cpu().numpy().squeeze(),
                #     '05_conv2_maxpool2d': activation['features.5'].cpu().numpy().squeeze(),
                #     '06_conv3_conv2d': activation['features.6'].cpu().numpy().squeeze(),
                #     '07_conv3_relu': activation['features.7'].cpu().numpy().squeeze(),
                #     '08_conv4_conv2d': activation['features.8'].cpu().numpy().squeeze(),
                #     '09_conv4_relu': activation['features.9'].cpu().numpy().squeeze(),
                #     '10_conv5_conv2d': activation['features.10'].cpu().numpy().squeeze(),
                #     '11_conv5_relu': activation['features.11'].cpu().numpy().squeeze(),
                #     '12_conv5_maxpool2d': activation['features.12'].cpu().numpy().squeeze(),
                #     '13_avgpool': activation['avgpool'].cpu().numpy().squeeze(),
                #     '14_fc6_dropout': activation['classifier.0'].cpu().numpy().squeeze(),
                #     '15_fc6_linear': activation['classifier.1'].cpu().numpy().squeeze(),
                #     '16_fc6_relu': activation['classifier.2'].cpu().numpy().squeeze(),
                #     '17_fc7_dropout': activation['classifier.3'].cpu().numpy().squeeze(),
                #     '18_fc7_linear': activation['classifier.4'].cpu().numpy().squeeze(),
                #     '19_fc7_relu': activation['classifier.5'].cpu().numpy().squeeze(),
                #     '20_fc8_linear': activation['classifier.6'].cpu().numpy().squeeze(),
                # }
                assets[ii]['modvals']['alexnet']['bg_w'] = get_modvals_alexnet()

                activation = {}
                tensor = preprocess(image_224_gray)
                input_batch = tensor.unsqueeze(0)
                input_batch = input_batch.to(device)
                alexnet.to(device)
                with torch.no_grad():
                    output = alexnet(input_batch)
                assets[ii]['modvals']['alexnet']['bg_g'] = get_modvals_alexnet()

                activation = {}
                tensor = preprocess(image_224_pinkn)
                input_batch = tensor.unsqueeze(0)
                input_batch = input_batch.to(device)
                alexnet.to(device)
                with torch.no_grad():
                    output = alexnet(input_batch)
                assets[ii]['modvals']['alexnet']['bg_g'] = get_modvals_alexnet()

        else:
            assets[ii]['transparency'] = False
            warn('No transparency found for {} and undecided how to handle, skipping for now...'.format(ii))
            continue
            # image = image.convert('RGB')
            # if image.size[0] != image.size[1]:
            #     image = ImageOps.pad(image, size=(np.max(image.size[:2]),) * 2, color=(255,) * 3)
            #
            # if 'modvals' not in assets[ii]:
            #     assets[ii]['modvals'] = {}



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

    if asset_counter % 100 == 0 and asset_counter > 0:
        save_snapshot(savestr='_asset_modvals_partial')
    asset_counter += 1
save_snapshot(savestr='_asset_modvals_full')


# Close pinknoise background images
background_224_pinkn.close()
background_299_pinkn.close()




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