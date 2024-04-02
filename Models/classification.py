from torchvision import models, transforms, datasets
from torchvision.utils import save_image, make_grid
import torch
import argparse
import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import os

os.makedirs("./Models/saved_models/classifier", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=20, help="Batch sizes")
parser.add_argument("--n_workers", type=int, default=4, help="Number of cpu workers")
parser.add_argument("--img_size", type=int, default=256, help="Image size (384 is standard for EfficientNet)")
parser.add_argument("--sample", type=int, default=3, help="Interval between validation sampling and model saving")
opt = parser.parse_args()

root = "./Datasets/Data (GAN)"
device = "mps"
if torch.cuda.is_available():
    device = "cuda"

## ----------------------------------------------------------------------------------------------------------
# DATA
## ----------------------------------------------------------------------------------------------------------

transform_ = transforms.Compose([
    transforms.Resize((opt.img_size, opt.img_size), interpolation = transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(), # transforms to [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard for B0
])

dataset = datasets.ImageFolder(root, transform=transform_)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

## ----------------------------------------------------------------------------------------------------------
# MODEL
## ----------------------------------------------------------------------------------------------------------

model = models.efficientnet_b0(weights= models.EfficientNet_B0_Weights.DEFAULT)
# model = models.efficientnet_b4(weights= models.EfficientNet_B4_Weights.DEFAULT)

n_classes = len(dataset.classes)
num_features = model.classifier[1].in_features  # Rewrite the classifier layer of the model
model.classifier[1] = nn.Linear(num_features, n_classes) 

loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

if device == "cuda" or device == "mps":
    model.to(device=device)
    loss_fn.to(device=device)
    # optimiser.to(device=device)

## ----------------------------------------------------------------------------------------------------------
# TRAIN
## ----------------------------------------------------------------------------------------------------------

def train(loader, epoch, optimiser, loss_fn):
    loop = tqdm.tqdm(loader)
    model.train()
    loop.set_description(f"Epoch {epoch}/{opt.n_epochs}")

    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        
        output = model(data)
        
        loss = loss_fn(output, targets)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        loop.set_postfix(loss = loss.item())

## ----------------------------------------------------------------------------------------------------------
# EVALUATION
## ----------------------------------------------------------------------------------------------------------
            
def evaluate(loader):
    loop = tqdm.tqdm(loader)
    model.eval()
    divider = 0

    for idx, (data, targets) in enumerate(loop):
        divider += len(data)
        data, targets = data.to(device), targets.to(device)
        with torch.no_grad():
            output = model(data)
        
        accuracy += sum(output == targets)
    return accuracy/divider


def sample_images(laoder):
    pass

if __name__ == "__main__":
    PATH = "./Models/saved_models/classifier/EfficientNetB0"

    for epoch in range(opt.n_epochs):
        train(dataloader, epoch, optimiser, loss_fn)
        if epoch % opt.sample == 0:
            accuracy = evaluate(dataloader)
            sample_images(dataloader)
            torch.save(model.state_dict(), PATH + f"_{epoch}.pt")
            
