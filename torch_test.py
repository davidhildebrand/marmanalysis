# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:45:17 2024

@author: DavidH
"""

import filetype
import os
from PIL import Image
import socket
import torch


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
if 'Galactica' in system_name:
    base_path = r'/Users/davidh/Data/Freiwald/'
    stim_path = r'/Users/davidh/Sync/Freiwald/MarmoScope/Stimulus/Sets'
    collated_stim_path = os.path.join(base_path, 'stims')
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
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

image_files = [f for f in os.listdir(collated_stim_path)
               if os.path.isfile(os.path.join(collated_stim_path, f))
               and filetype.is_image(os.path.join(collated_stim_path, f))]

for imf in image_files:
    filename = os.path.join(collated_stim_path, imf)
    with Image.open(filename) as img:
        if has_transparency(img):
            print(f'{imf} has transparency')
        else:
            print(f'{imf} does not have transparency')
        # img.show()


model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
# model.eval()
fc6 = model.classifier[1]  # want output from 'fc6' (1): Linear(in_features=9216, out_features=4096, bias=True)
fc6.register_forward_hook(get_activation('fc6'))


for imf in image_files:


from PIL import Image
from torchvision import transforms
input_image = Image.open(filename).convert('RGBA')
newbg = Image.new('RGBA', input_image.size, (128, 128, 128))
ac = Image.alpha_composite(newbg, input_image)
input_image = ac.convert('RGB')
preprocess = transforms.Compose([
    # transforms.Resize(256),
    # transforms.Resize(224),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

# transforms.functional.to_pil_image(input_tensor)


# move the input and model to GPU for speed if available
if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)
# Tensor of shape 1000, with confidence scores over ImageNet's 1000 classes
print(output[0])
# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
probabilities = torch.nn.functional.softmax(output[0], dim=0)
print(probabilities)

# # Download ImageNet labels
# !wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

# Read the categories
with open(r"F:\Data\stimuli\imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
# Show top categories per image
top5_prob, top5_catid = torch.topk(probabilities, 5)
for i in range(top5_prob.size(0)):
    print(categories[top5_catid[i]], top5_prob[i].item())