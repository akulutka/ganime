import datetime
import numpy as np
import os
import pickle
from PIL import Image
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

class Anime_Dataset:
    def __init__(self, root, class_num, transform):
        self.root = root
        self.img_folder = os.path.join(self.root, 'images')
        self.label_file = os.path.join(self.root, 'labels.pkl')
        self.img_files = os.listdir(self.img_folder)
        self.labels = pickle.load(open(self.label_file, 'rb'))
        self.preprocess()
        self.class_num = class_num
        self.transform = transform
        
        assert(len(self.img_files) <= len(self.labels))
    
    def preprocess(self):
        new_label = {}
        for img, tag in self.labels.items():
            if tag[-1] is None:
                new_label[img] = tag[:-1]
        self.labels = new_label
        self.img_files = [path for path in self.img_files if os.path.splitext(path)[0] in self.labels]
        print(len(self.labels), len(self.img_files))
    
    def color_transform(self, x):
        x = torchvision.transforms.functional.adjust_saturation(x, 2.75)
        x = torchvision.transforms.functional.adjust_gamma(x, 0.7)
        x = torchvision.transforms.functional.adjust_contrast(x, 0.8)
        x = torchvision.transforms.functional.adjust_brightness(x, 0.75)
        return x
        
    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_folder, self.img_files[idx]))
        img = self.color_transform(img)
        img = self.transform(img)
        filename = os.path.splitext(self.img_files[idx])[0]
        label = self.labels[filename]
        
        one_hots = []
        mask = []
        for i, c in enumerate(self.class_num):
            l = torch.zeros(c)
            m = torch.zeros(c)
            if label[i]:
                l[label[i]] = 1
                m = 1 - m # create mask
            one_hots.append(l)
            mask.append(m)
        one_hots = torch.cat(one_hots, 0)
        mask = torch.cat(mask, 0)
        return img, one_hots, mask

def get_anime_dataloader(root, classes, batch_size):
    
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = Anime_Dataset(root, classes, transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, drop_last = True)
    return train_loader, test_loader
    
def denorm(img):
    """ Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    """

    output = img / 2 + 0.5
    return output.clamp(0, 1)


MODEL_NAME = 'R1'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_DIM = (64, 64, 3)


def tensor2img(tensor):
    img = (np.transpose(tensor.detach().cpu().numpy(), [1,2,0])+1)/2.
    return img


class ResidualBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, groups=1):
        super(ResidualBlock, self).__init__()
        p = kernel_size//2
        self.conv1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=p),
            nn.LeakyReLU(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size, padding=p),
            nn.LeakyReLU(0.2)
        )
        self.proj = nn.Conv2d(inplanes, planes, 1) if inplanes != planes else None
    
    def forward(self, x):
        identity = x
        
        y = self.conv1(x)
        y = self.conv2(y)
        
        identity = identity if self.proj is None else self.proj(identity)
        y = y + identity
        return y


class Discriminator(nn.Module):
    """
        Convolutional Discriminator
    """
    def __init__(self, in_channel=1):
        super(Discriminator, self).__init__()
        self.D = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1), # (N, 64, 64, 64)
            ResidualBlock(64, 128),
            nn.AvgPool2d(3, 2, padding=1), # (N, 128, 32, 32)
            ResidualBlock(128, 256),
            nn.AvgPool2d(3, 2, padding=1), # (N, 256, 16, 16)
            ResidualBlock(256, 512),
            nn.AvgPool2d(3, 2, padding=1), # (N, 512, 8, 8)
            ResidualBlock(512, 1024),
            nn.AvgPool2d(3, 2, padding=1) # (N, 1024, 4, 4)
        )
        self.fc = nn.Linear(1024*4*4, 1) # (N, 1)
        
    def forward(self, x):
        B = x.size(0)
        h = self.D(x)
        h = h.view(B, -1)
        y = self.fc(h)
        return y


class Generator(nn.Module):
    """
        Convolutional Generator
    """
    def __init__(self, out_channel=1, n_filters=128, n_noise=512):
        super(Generator, self).__init__()
        self.fc = nn.Linear(n_noise, 1024*4*4)
        self.G = nn.Sequential(
            ResidualBlock(1024, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (N, 512, 8, 8)
            ResidualBlock(512, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (N, 256, 16, 16)
            ResidualBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (N, 128, 32, 32)
            ResidualBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # (N, 64, 64, 64)
            ResidualBlock(64, 64),
            nn.Conv2d(64, out_channel, 3, padding=1) # (N, 3, 64, 64)
        )
        
    def forward(self, z):
        B = z.size(0)
        h = self.fc(z)
        h = h.view(B, 1024, 4, 4)
        x = self.G(h)
        return x


transform = transforms.Compose([transforms.Resize((IMAGE_DIM[0],IMAGE_DIM[1])),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                std=(0.5, 0.5, 0.5))
                               ]
)

batch_size = 64
n_noise = 256
data_loader, _ = get_anime_dataloader('./data', (10, 12), 64)
D = Discriminator(in_channel=IMAGE_DIM[-1]).to(DEVICE)
G = Generator(out_channel=IMAGE_DIM[-1], n_noise=n_noise).to(DEVICE)
D_opt = torch.optim.RMSprop(D.parameters(), lr=1e-4, alpha=0.99)
G_opt = torch.optim.RMSprop(G.parameters(), lr=1e-4, alpha=0.99)
n_gpu = 4

def r1loss(inputs, label=None):
    # non-saturating loss with R1 regularization
    l = -1 if label else 1
    return F.softplus(l*inputs).mean()

max_epoch = 101
step = 0
log_term = 1000
save_term = 1000
upd_term = 400
r1_gamma = 10
steps_per_epoch = len(data_loader.dataset) // batch_size
for epoch in range(max_epoch):
    for idx, data in enumerate(tqdm(data_loader, total=len(data_loader))):
        G.zero_grad()
        images = data[0]
        # Training Discriminator
        x = images.to(DEVICE)
        x.requires_grad = True
        x_outputs = D(x)
        d_real_loss = r1loss(x_outputs, True)
        # Reference >> https://github.com/rosinality/style-based-gan-pytorch/blob/a3d000e707b70d1a5fc277912dc9d7432d6e6069/train.py
        # little different with original DiracGAN
        grad_real = grad(outputs=x_outputs.sum(), inputs=x, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2).mean()
        grad_penalty = 0.5*r1_gamma*grad_penalty
        D_x_loss = d_real_loss + grad_penalty

        z = (torch.rand(size=[batch_size, n_noise])*2-1).to(DEVICE)
        x_fake = G(z)
        z_outputs = D(x_fake.detach())
        D_z_loss = r1loss(z_outputs, False)
        D_loss = D_x_loss + D_z_loss
        
        D.zero_grad()
        D_loss.backward()
        D_opt.step()

        # Training Generator
        z = (torch.rand(size=[batch_size, n_noise])*2-1).to(DEVICE)
        x_fake = G(z)
        z_outputs = D(x_fake)
        G_loss = r1loss(z_outputs, True)
        
        G.zero_grad()
        G_loss.backward()
        G_opt.step()
        
        if step % log_term == 0:
            dt = datetime.datetime.now().strftime('%H:%M:%S')
            print('Epoch: {}/{}, Step: {}, D Loss: {:.4f}, G Loss: {:.4f}, gp: {:.4f}, Time:{}'.format(epoch, max_epoch, step, D_loss.item(), G_loss.item(), grad_penalty.item(), dt))
        
        step += 1

torch.save(G.state_dict(), "./state_dicts/r1gan_aniG.w")
torch.save(D.state_dict(), "./state_dicts/r1gan_aniD.w")