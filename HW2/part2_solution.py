# Don't erase the template code, except "Your code here" comments.

import torch
# Your code here...
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision
from PIL import Image
from IPython.display import Image
from tqdm import trange, tqdm
#import wandb
from time import time


def get_dataloader(path, kind):
    """
    Return dataloader for a `kind` split of Tiny ImageNet.
    If `kind` is 'val', the dataloader should be deterministic.
    
    path:
        `str`
        Path to the dataset root - a directory which contains 'train' and 'val' folders.
    kind:
        `str`
        'train' or 'val'
        
    return:
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        For each batch, should yield a tuple `(preprocessed_images, labels)` where
        `preprocessed_images` is a proper input for `predict()` and `labels` is a
        `torch.int64` tensor of shape `(batch_size,)` with ground truth class labels.
    """
    # Your code here
    batch_size = 100
    
    if kind == 'train':
        train_transform = transforms.Compose([
                transforms.RandomRotation(5),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees = 10, shear = 10, scale = (0.7, 1.05)),
                transforms.ToTensor()])
        
        data_train = torchvision.datasets.ImageFolder(path+'train', transform = train_transform)
        train_loader = torch.utils.data.DataLoader(data_train, batch_size=batch_size,shuffle=True)
        
        return train_loader
        
    elif kind == 'val':
        val_transform = transforms.Compose([
                transforms.ToTensor()]) 
        data_val = torchvision.datasets.ImageFolder(path+'val', transform = val_transform)
        val_loader = torch.utils.data.DataLoader(data_val, batch_size = batch_size)
        
        return val_loader
    

def get_model():
    """
    Create neural net object, initialize it with raw weights, upload it to GPU.
    
    return:
    model:
        `torch.nn.Module`
    """
    # Your code here

    class DenseNet(torch.nn.Module):
        def __init__(self):
            super(DenseNet, self).__init__()
    
            self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.batn1 = torch.nn.BatchNorm2d(32)
            self.act1  = torch.nn.ReLU()
    
            self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1) 
            self.batn2 = torch.nn.BatchNorm2d(32)
            self.act2  = torch.nn.ReLU()
    
            self.conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.batn3 = torch.nn.BatchNorm2d(64)
            self.act3  = torch.nn.ReLU()
    
            #CAT
            self.pool1 = torch.nn.MaxPool2d(kernel_size=2)
            self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.batn4 = torch.nn.BatchNorm2d(128)
            self.act4  = torch.nn.ReLU()
    
            self.conv5 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.batn5 = torch.nn.BatchNorm2d(256)
            self.act5  = torch.nn.ReLU()
    
            #CAT
            self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
            self.conv6 = torch.nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
            self.batn6 = torch.nn.BatchNorm2d(256)
            self.act6  = torch.nn.ReLU()
    
            self.conv7 = torch.nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
            self.batn7 = torch.nn.BatchNorm2d(128)
            self.act7  = torch.nn.ReLU()
            self.pool3 = torch.nn.AvgPool2d(kernel_size=2)
    
            self.flat = torch.nn.Flatten()
            self.fc1   = torch.nn.Linear(2048,200)
            self.act8 = torch.nn.LogSoftmax(dim=1)
            
        def forward(self,x):
            x = self.conv1(x)
            x = self.batn1(x)
            x = self.act1(x)
            a1 = x 
    
            x = self.conv2(x)
            x = self.batn2(x)
            x = self.act2(x)
            a2 = x
    
            x = self.conv3(x)
            x = self.batn3(x)
            x = self.act3(x)
    
            x = torch.cat((a1,a2,x), dim = 1)
    
            x = self.pool1(x)
            x = self.conv4(x)
            x = self.batn4(x)
            x = self.act4(x)
            b1 = x 
    
            x = self.conv5(x)
            x = self.batn5(x)
            x = self.act5(x)
    
            x = torch.cat((b1,x), dim = 1)
    
            x = self.pool2(x)
            x = self.conv6(x)
            x = self.batn6(x)
            x = self.act6(x)
    
            x = self.conv7(x)
            x = self.batn7(x)
            x = self.act7(x)
            x = self.pool3(x)
    
            x = self.flat(x)
            x = self.fc1(x)
            x = self.act8(x)
    
            return x   
    
    model = DenseNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model


def get_optimizer(model):
    """
    Create an optimizer object for `model`, tuned for `train_on_tinyimagenet()`.
    
    return:
    optimizer:
        `torch.optim.Optimizer`
    """
    # Your code here
    adam_optimizer = torch.optim.Adam(model.parameters(), weight_decay = 1e-4)
    return adam_optimizer

def predict(model, batch):
    """
    model:
        `torch.nn.Module`
        The neural net, as defined by `get_model()`.
    batch:
        unspecified
        A batch of Tiny ImageNet images, as yielded by `get_dataloader(..., 'val')`
        (with same preprocessing and device).
    
    return:
    prediction:
        `torch.tensor`, shape == (N, 200), dtype == `torch.float32`
        The scores of each input image to belong to each of the dataset classes.
        Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
        belong to `j`-th class.
        These scores can be 0..1 probabilities, but for better numerical stability
        they can also be raw class scores after the last (usually linear) layer,
        i.e. BEFORE softmax.
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch = batch.to(device)
    preds = model.forward(batch)
    return preds

def validate(dataloader, model):
    """
    Run `model` through all samples in `dataloader`, compute accuracy and loss.
    
    dataloader:
        `torch.utils.data.DataLoader` or an object with equivalent interface
        See `get_dataloader()`.
    model:
        `torch.nn.Module`
        See `get_model()`.

    return:
    accuracy:
        `float`
        The fraction of samples from `dataloader` correctly classified by `model`
        (top-1 accuracy). `0.0 <= accuracy <= 1.0`
    loss:
        `float`
        Average loss over all `dataloader` samples.
    """
    # Your code here
    criterion = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    val_loss = torch.tensor([0.0]).to(device)
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)  
            y = y.to(device)  
            probs = model(x)
            preds = probs.max(axis = 1)[1]
            correct += (preds == y).sum().item()
            total += len(y)
            val_loss += criterion(probs, y)
    val_accuracy = correct/total
    val_loss = val_loss/total
    return val_accuracy, val_loss

def train_on_tinyimagenet(train_dataloader, val_dataloader, model, optimizer):
    """
    Train `model` on `train_dataloader` using `optimizer`. Use best-accuracy settings.
    
    train_dataloader:
    val_dataloader:
        See `get_dataloader()`.
    model:
        See `get_model()`.
    optimizer:
        See `get_optimizer()`.
    """
    # Your code here
    def get_accuracy(model, dataloader, device):
        correct = 0
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(device)  
                y = y.to(device)  
                prediction = model(x).argmax(dim=-1, keepdim=True)
                correct += prediction.eq(y.view_as(prediction)).sum().item()
        return correct / len(dataloader.dataset)
    
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Starting Training using device : ', device)
        
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [20,30], gamma = 0.1)  
    n_epochs = 100
    
    max_val_score = 0
    max_val_score_epoch = 0
    
    first_epoch_loss = 0
    train_losses = []
    train_scores = []
    val_scores = []
    
    
    for epoch in trange(n_epochs):
            
        #wandb.log({'epoch': epoch})  
    
        model.train()
        current_loss = 0
        time_start = time()
        for x_batch, y_batch in train_dataloader:
            x = x_batch.to(device)
            y = y_batch.to(device)
            optimizer.zero_grad()
    
            preds = model(x)
            loss = criterion(preds, y)
    
            #wandb.log({'batch_loss': loss.item()})
            current_loss += loss.item()
    
            loss.backward()
            optimizer.step()
    
        #wandb.log({'Time':time() - time_start})
        if epoch == 0:
            first_epoch_loss = current_loss
        train_losses.append(current_loss/first_epoch_loss)
        #wandb.log({'loss':current_loss/first_epoch_loss})
    
        model.eval()
        train_score = get_accuracy(model, train_dataloader, device)
        val_score = get_accuracy(model, val_dataloader, device)
    
        if scheduler is not None:  
            scheduler.step()
    
    
        print("Validation accuracy:%.2f%%" % (val_score * 100))
    
        if val_score > max_val_score:
            max_val_score = val_score
            max_val_score_epoch = epoch
            torch.save(model.state_dict(), './'+'Trained'+'_best_params.pt')
            
        train_scores.append(train_score)
        val_scores.append(val_score)
        #wandb.log({'Train_accuracy': train_score * 100})
        #wandb.log({'Validation_accuracy': val_score* 100})


def load_weights(model, checkpoint_path):
    """
    Initialize `model`'s weights from `checkpoint_path` file.
    
    model:
        `torch.nn.Module`
        See `get_model()`.
    checkpoint_path:
        `str`
        Path to the checkpoint.
    """
    # Your code here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    model.eval()

    print('Loaded. Model is ready for evaluation')

def get_checkpoint_metadata():
    """
    Return hard-coded metadata for 'checkpoint.pth'.
    Very important for grading.

    return:
    md5_checksum:
        `str`
        MD5 checksum for the submitted 'checkpoint.pth'.
        On Linux (in Colab too), use `$ md5sum checkpoint.pth`.
        On Windows, use `> CertUtil -hashfile checkpoint.pth MD5`.
        On Mac, use `$ brew install md5sha1sum`.
    google_drive_link:
        `str`
        View-only Google Drive link to the submitted 'checkpoint.pth'.
        The file must have the same checksum as in `md5_checksum`.
    """
    # Your code here;
    md5_checksum = "6acd9a9ac8758dc0508b1775ba672516"
    # Your code here;
    google_drive_link = "https://drive.google.com/file/d/1HOGoYd7gLVlQ4-tHsKl5K0v0IwQULYRf/view?usp=sharing"

    return md5_checksum, google_drive_link