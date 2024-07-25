# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:45:17 2024

@author: DavidH
"""

# based on https://pytorch.org/hub/pytorch_vision_alexnet/

import torch
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()

# import urllib
# # url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# url, filename = ("https://images.pexels.com/photos/667500/pexels-photo-667500.jpeg", "c.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

filename = r'F:\Data\stimuli\FOBmany\Images\20240312d\FreiwaldFOB2012_Human_Head_10_erode3px.png'
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