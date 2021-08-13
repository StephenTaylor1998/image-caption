import torch
from torch import nn
import torchvision


class Caption(nn.Module):
    def __init__(self):
        super(Caption, self).__init__()
        self.model = nn.Sequential(*list(torchvision.models.resnet18(pretrained=True).children())[:-2])
        self.adaptiveAvgPool2d = nn.AdaptiveAvgPool2d(4)
        self.fc1 = nn.Linear(16, 10)
        self.fc2 = nn.Linear(512, 6728)

    def forward(self, x):
        x = self.model(x)
        x = self.adaptiveAvgPool2d(x)
        x = torch.reshape(x, (*x.shape[:2], x.shape[2] * x.shape[3]))
        x = self.fc1(x)
        x = torch.transpose(x, -1, -2)
        x = self.fc2(x)
        x = torch.transpose(x, -1, -2)
        return x



def resnet18_caption(**kwargs):
    return Caption()


if __name__ == '__main__':
    model = resnet18_caption()

    inp = torch.ones((1, 3, 512, 224))

    out = model(inp)

    print(out.shape)