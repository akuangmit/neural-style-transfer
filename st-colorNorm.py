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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128  

loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()])  

# Apply color on luminance image, and return rgb image.
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

############ Load the content and style images ##############

def image_loader(image_name, gray=False):
    image = loader(Image.open(image_name)).unsqueeze(0)
    if gray:
        rgb2yiq = np.array([[0.299,0.587,0.114],[0.596,-0.274,-0.322],[0.211,-0.523,0.312]])
        image = image.permute((2,3,1,0))
        image = np.matmul(rgb2yiq, image.numpy())
        image = torch.from_numpy(image).permute((3,2,0,1))
    return image.to(device, torch.float)

#Retrieves luminance channel of image
def L_channel(image): # tensor format
    tmp = image.clone()
    return tmp.clone().narrow(1,0,1)

#Retrieves color channels of image
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
# lol = yiq2rgb(content_img_gray, content_img_iq)

style_img_yiq = image_loader("./StyleImages/strokes.jpg",True)
style_img_gray = L_channel(style_img_yiq)
content_img_gray = L_normalize(style_img_gray, content_img_gray)
# norm_img_gray = L_normalize(content_img_gray, style_img_gray)
# print(style_img_gray.mean())

# seq = [style_img_gray.flatten().numpy(), norm_img_gray.flatten().numpy(), content_img_gray.flatten().numpy()]
# seqLabels = ['Autumn', 'Normalized Autumn', 'Stata']
# plt.hist(seq, label=seqLabels, density=True, stacked=True)
# plt.title('Intensity histograms')
# plt.legend()
# plt.show()
assert style_img_gray.size() == content_img_gray.size(), \
    "we need to import style and content images of the same size"


unloader = transforms.ToPILImage()  

plt.ion()

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  
    image = image.squeeze(0)      
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) 

# plt.figure()
# imshow(style_img_gray, title='Style Image')

# plt.figure()
# imshow(content_img_gray, title='Content Image')

############## Loss Modules ##########################
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

def gram_matrix(input):
    a, b, c, d = input.size()  

    features = input.view(a * b, c * d)  

    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

 
########### Import VGG Model ##################
cnn = models.vgg19(pretrained=True).features.to(device).eval()



cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

# The convolution layers to insert the loss modules after
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

##### Create new model that is VGG + Loss Modules #########
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0 
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

############# Initial output image initialization ####################
input_img = content_img_gray.clone()
#input_img = torch.randn(content_img.data.size(), device=device)

#plt.figure()
#imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

######## Style Transfer Procedure ##############
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

################ The call to run style transfer code ######################
output_gray = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img_gray, style_img_gray, input_img) #Apply style transfer on luminance-only images #Return final output as colors of content image applied to luminance-only image
output = yiq2rgb(output_gray.data, content_img_iq) #Return final output as colors of content image applied to luminance-only image

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()
