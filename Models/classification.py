from torchvision import models, transforms, datasets
import torch
import argparse
import tqdm
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=200, help="Number of epochs for training")
parser.add_argument("--batch_size", type=int, default=20, help="Batch sizes")
parser.add_argument("--n_workers", type=int, default=0, help="Number of cpu workers")
opt = parser.parse_args()

root = "./Datasets/Data (GAN)"

if torch.cuda.is_available():
    device = "cuda"

## ----------------------------------------------------------------------------------------------------------
# DATA
## ----------------------------------------------------------------------------------------------------------

transform_ = transforms.Compose([
    transforms.Resize(255),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(root, transform=transform_)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

## ----------------------------------------------------------------------------------------------------------
# MODEL
## ----------------------------------------------------------------------------------------------------------

model = models.efficientnet_b4(weights = models.EfficientNet_B4_Weights)
loss_fn = torch.nn.CrossEntropyLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=0.001)

if torch.cuda.is_available():
    model.to(device=device)
    loss_fn.to(device=device)
    optimiser.to(device=device)

## ----------------------------------------------------------------------------------------------------------
# TRAIN
## ----------------------------------------------------------------------------------------------------------

def train(loader, epochs, optimiser, loss_fn):
    loop = tqdm.tqdm(loader)
    model.train()

    for epoch in range(0, epochs):
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

    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device), targets.to(device)
        output = model(data)


if __name__ == "__main__":
    train(dataloader, optimiser, loss_fn)
