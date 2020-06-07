from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy
import numpy as np


######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device. 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be between
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images. 
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <https://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <https://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

def yiq2rgb(L, IQ):
    '''
    input: 1x3x128x128 tensor in yiq
    return: 1x3x128x128 tensor in rgb
    '''
    liq = torch.cat((L,IQ),1)
    print(liq.shape) # 1x3x128x128

    rgb2yiq_inv = np.array([[1, 0.956, 0.621],[1, -0.273, -0.647],[1,-1.104, 1.7]],dtype=np.float32)
    image = liq.permute((2,3,1,0))
    image = np.matmul(rgb2yiq_inv, image.numpy())
    image = torch.from_numpy(image).permute((3,2,0,1))
    return image

def image_loader(image_name, gray=False):
    image = loader(Image.open(image_name)).unsqueeze(0)
    if gray:
        rgb2yiq = np.array([[0.299,0.587,0.114],[0.596,-0.274,-0.322],[0.211,-0.523,0.312]])
        image = image.permute((2,3,1,0))
        image = np.matmul(rgb2yiq, image.numpy())
        image = torch.from_numpy(image).permute((3,2,0,1))
    return image.to(device, torch.float)

def L_channel(image): # tensor format
    tmp = image.clone()
    return tmp.clone().narrow(1,0,1)

def IQ_channel(image):  # tensor format
    tmp = image.clone()
    return tmp.clone().narrow(1,1,2)

def L_normalize(content, color):
    contentMu = content.mean()
    contentStd = content.std()

    colorMu = color.mean()
    colorStd = color.std()
    return (colorStd/contentStd)*(color.clone()-colorMu)+contentMu

content_img_yiq = image_loader("./ContentImages/waterfall.jpg", True)
content_img_gray = L_channel(content_img_yiq)
content_img_iq = IQ_channel(content_img_yiq)

color_img_yiq = image_loader("./StyleImages/strokes.jpg", True)
color_img_gray = L_channel(color_img_yiq)
color_img_iq = IQ_channel(color_img_yiq)
norm_color_gray = L_normalize(content_img_gray, color_img_gray)
# print(content_img_gray.mean())
# print(norm_color_gray.mean())

style_img_yiq = image_loader("./StyleImages/strokes.jpg",True)
style_img_gray = L_channel(style_img_yiq)

# plt.plot(torch.histc(color_img_gray).numpy(), label='Autumn')
plt.plot(torch.histc(content_img_gray).numpy(), label='Waterfall (Content)')
# plt.plot(torch.histc(norm_color_gray).numpy(), label='Norm. Strokes (Style)')
plt.plot(torch.histc(style_img_gray).numpy(), label='Strokes (Style)')
plt.title('Luminance Intensity Histograms')
plt.legend()
plt.show()
assert style_img_gray.size() == content_img_gray.size(), \
    "we need to import style and content images of the same size"


######################################################################
# Now, let's create a function that displays an image by reconverting a 
# copy of it to PIL format and displaying the copy using 
# ``plt.imshow``. We will try displaying the content and style images 
# to ensure they were imported correctly.

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated

# plt.figure()
# lol = yiq2rgb(color_img_gray, color_img_iq)
# imshow(lol, title='Color Image')

# plt.figure()
# lol = yiq2rgb(norm_color_gray, color_img_iq)
# imshow(lol, title='Norm Image')
# plt.show()

import time
time.sleep(10)

