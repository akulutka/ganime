import os
import pickle
from PIL import Image, ImageFile
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

pic_width = 64

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
        x = F.adjust_saturation(x, 2.5)
        x = F.adjust_gamma(x, 0.7)
        x = F.adjust_contrast(x, 1.2)
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
        transforms.Resize((pic_width, pic_width)),
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
    ''' Denormalize input image tensor. (From [0,1] -> [-1,1]) 
    
    Args:
        img: input image tensor.
    '''
	
    output = img / 2 + 0.5
    return output.clamp(0, 1)

train_loader, test_loader = get_anime_dataloader('./data', (10, 12), 64)

eye_labels = [
    'aqua',
    'black',
    'blue',
    'brown',
    'green',
    'orange',
    'pink',
    'purple',
    'red',
    'yellow'
]

hair_labels = [
    'aqua',
    'black',
    'blonde',
    'blue',
    'brown',
    'gray',
    'green',
    'orange',
    'pink',
    'purple',
    'red',
    'white'
]

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(10368, 625), # 4 * 4 * 128
            nn.BatchNorm1d(625),
            nn.ReLU(),
            nn.Linear(625, 22)
        )        
    
    def forward(self, x):
        out = self.layer1(x)
        out = out.view(out.size(0), -1)
        return self.layer2(out)


model = ConvNet()
opt = torch.optim.Adam(model.parameters(), lr=0.005)
loss_func = nn.CrossEntropyLoss()
losses = []
accs = []
n_epoch = 25
model.to(device)

for epoch in range(n_epoch):
    for i, (batch_X, batch_Y, _) in enumerate(train_loader):
        model.train()
        batch_Y_eye = torch.argmax(batch_Y[:, :10], dim=1)
        batch_Y_hair = torch.argmax(batch_Y[:, 10:], dim=1)
        X = Variable(batch_X.to(device))
        Y_eye = Variable(batch_Y_eye.to(device))
        Y_hair = Variable(batch_Y_hair.to(device))
        opt.zero_grad()
        Y_pred = model(X)
        Y_pred_eye = Y_pred[:, :10]
        Y_pred_hair = Y_pred[:, 10:]
        loss = loss_func(Y_pred_eye, Y_eye) + loss_func(Y_pred_hair, Y_hair)
        losses.append(loss)
        accs.append(((Y_pred_eye.argmax(dim=1) == torch.tensor(Y_eye)).float().mean() + (Y_pred_hair.argmax(dim=1) == torch.tensor(Y_hair)).float().mean()) / 2)
        loss.backward()
        opt.step()


test_accs = []

for i, (batch_X, batch_Y, _) in enumerate(test_loader):
    model.eval()
    batch_Y_eye = torch.argmax(batch_Y[:, :10], dim=1)
    batch_Y_hair = torch.argmax(batch_Y[:, 10:], dim=1)
    X = Variable(batch_X.to(device))
    Y_eye = Variable(batch_Y_eye.to(device))
    Y_hair = Variable(batch_Y_hair.to(device))
    Y_pred = model(X)
    Y_pred_eye = Y_pred[:, :10]
    Y_pred_hair = Y_pred[:, 10:]
    test_accs.append(((Y_pred_eye.argmax(dim=1) == torch.tensor(Y_eye)).float().mean() + (Y_pred_hair.argmax(dim=1) == torch.tensor(Y_hair)).float().mean()) / 2)
    print(f'Accuracy on test dataset: {torch.tensor(test_accs).mean() * 100}%')

torch.save(model.state_dict(), './state_dict.w')