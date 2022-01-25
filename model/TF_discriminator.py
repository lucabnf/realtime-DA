import torch.nn as nn
import torch.nn.functional as F
import torch
import tltorch


class TF_Discriminator(nn.Module):

  def __init__(self, num_classes, ndf = 64):
    super(TF_Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
    self.fact_conv1 = tltorch.FactorizedConv.from_conv(self.conv1, rank=0.5, decompose_weights=True, factorization='tucker')
    #self.fact_conv1 = tltorch.FactorizedConv(num_classes, ndf, kernel_size=4, stride=2, padding=1, order = 2, rank='same', factorization='cp')

    self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
    self.fact_conv2 = tltorch.FactorizedConv.from_conv(self.conv2, rank=0.5, decompose_weights=True, factorization='tucker')
    #self.fact_conv2 = tltorch.FactorizedConv(ndf, ndf*2, kernel_size=4, stride=2, padding=1, order = 2, rank='same', factorization='cp')

    self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
    self.fact_conv3 = tltorch.FactorizedConv.from_conv(self.conv3, rank=0.5, decompose_weights=True, factorization='tucker')
    #self.fact_conv3 = tltorch.FactorizedConv(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, order = 2 ,rank='same', factorization='cp')

    self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
    self.fact_conv4 = tltorch.FactorizedConv.from_conv(self.conv4, rank=0.5, decompose_weights=True, factorization='tucker')
    #self.fact_conv4 = tltorch.FactorizedConv(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, order = 2, rank='same', factorization='cp')

    self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)
    self.fact_classifier = tltorch.FactorizedConv.from_conv(self.classifier, rank=0.5, decompose_weights=True, factorization='tucker')
    #self.fact_classifier = tltorch.FactorizedConv(ndf*8, 1, kernel_size=4, stride=2, padding=1,order = 2, rank='same', factorization='cp')

    self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

  def forward(self, x):
    x = self.fact_conv1(x)
    x = self.leaky_relu(x)

    x = self.fact_conv2(x)
    x = self.leaky_relu(x)

    x = self.fact_conv3(x)
    x = self.leaky_relu(x)

    x = self.fact_conv4(x)
    x = self.leaky_relu(x)
    
    x = self.fact_classifier(x)

    return x