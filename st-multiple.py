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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imsize = 512 if torch.cuda.is_available() else 128  

loader = transforms.Compose([
    transforms.Resize(imsize),  
    transforms.ToTensor()])  

############ Load the content and style images ##############
def image_loader(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

content_img = image_loader("./ContentImages/dancing.jpg")
style_img = image_loader("./StyleImages/autumn.jpg")
style_img2 = image_loader("./StyleImages/starrynight.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"
assert style_img2.size() == content_img.size(), \
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
# imshow(style_img, title='Style Image')

plt.figure()
imshow(content_img, title='Content Image')
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

###### Define additional style loss module for the second style
class StyleLoss2(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss2, self).__init__()
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
                               style_img, content_img, style_img2, 
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)


    content_losses = []
    style_losses = []
    style_losses2 = []

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

            #Add the second style loss module
            target_feature2 = model(style_img2).detach()
            style_loss2 = StyleLoss2(target_feature2)
            model.add_module("style_loss2_{}".format(i), style_loss2)
            style_losses2.append(style_loss2)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], StyleLoss2):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses, style_losses2

############# Initial output image initialization ####################
input_img = content_img.clone()
# input_img = torch.randn(content_img.data.size(), device=device)

# plt.figure()
# imshow(input_img, title='Input Image')


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer

######## Style Transfer Procedure: Involves two style weights for the two styles ##############
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, style_img2, input_img, num_steps=300,
                       style_weight=300000, style_weight2=700000, content_weight=30): 
    print('Building the style transfer model..')
    model, style_losses, content_losses, style_losses2 = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img, style_img2)
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            style_score2 = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for sl2 in style_losses2:
                style_score2 += sl2.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            style_score2 *= style_weight2
            content_score *= content_weight

            loss = style_score + style_score2 + content_score
            loss.backward(retain_graph = True)

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Style Loss2: {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), style_score2.item(), content_score.item()))
                print()

            return style_score + style_score2 + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img

################ The call to run style transfer code ######################
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, style_img2, input_img)

plt.figure()
imshow(output, title='Output Image')

plt.ioff()
plt.show()

