import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SimpleCNN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.resnet = models.resnet34(weights=models.resnet.ResNet34_Weights.DEFAULT)
        if in_dim != 3:
            ori_conv1 = self.resnet.conv1
            self.resnet.conv1 = nn.Conv2d(in_channels=in_dim, out_channels=ori_conv1.out_channels,
                                          kernel_size=ori_conv1.kernel_size,
                                          stride=ori_conv1.stride, padding=ori_conv1.padding)
        feature_dim = self.resnet.fc.out_features

        self.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)

    def forward(self, inputs):
        feature = self.resnet(inputs)
        logits = self.fc(feature)
        prob = F.softmax(logits, dim=1)

        return feature, logits, prob

    def criterion(self, feature, logits, label):
        pred = torch.argmax(logits, dim=1)
        target = torch.argmax(label, dim=1)

        loss_ce = F.cross_entropy(logits, target)
        return loss_ce

class SimpleENN(SimpleCNN):
    def __init__(self, in_dim, out_dim, focal, alpha_kl):
        super(SimpleENN, self).__init__(in_dim, out_dim)
        self.alpha_kl = alpha_kl
        self.focal = focal

    def forward(self, inputs):
        feature = self.resnet(inputs)
        evidence = torch.exp(self.fc(feature))
        prob = F.normalize(evidence + 1, p=1, dim=1)

        return feature, evidence, prob

    def criterion(self, feature, evidence, label):
        """
        evicential cross_entropy for ENN
        """
        alpha = evidence + 1
        prob = F.normalize(alpha, dim=1)

        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, self.out_dim)

        loss_ece = torch.sum(label * (torch.digamma(alpha_0) - torch.digamma(alpha)), dim=1)
        loss_ece = torch.mean(loss_ece)

        loss_kl = self.regular_kl_dist(feature, evidence, label)
        # loss_kl = 0
        return loss_ece + self.alpha_kl * loss_kl

    def regular_kl_dist(self, feature, evidence, label):
        # evidence[torch.where(evidence > 100)] = 100
        alpha = evidence + 1
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        beta = torch.ones((1, evidence.shape[-1])).to(evidence.device.type)
        # Mbeta = torch.ones((alpha.shape[0],c)).cuda()
        S_beta = torch.sum(beta, dim=1, keepdim=True)
        lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
        dg0 = torch.digamma(S_alpha)
        dg1 = torch.digamma(alpha)
        kl = torch.mean(torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni)
        return kl

# MENN is ERNN
class EMENN(SimpleENN):
    def forward(self, inputs):
        feature = self.resnet(inputs)
        evidence = torch.exp(self.fc(feature))

        em_evidence = evidence - torch.min(evidence, dim=1, keepdim=True).values
        prob = F.normalize(em_evidence + 1, p=1, dim=1)
        return feature, em_evidence, prob

