import torchvision
import torch
import torchvision.models.segmentation as models
import torch.utils.data as data
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from imageload import ImageLoad
import numpy as np
import tqdm

## IMPORT
MASK_DATA_PATH = "./data/Ground Truth/"
IMAGE_DATA_PATH = "./data/Original/"
#mask_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG_TRAIN)

## DATA
batch_size_train = 32
batch_size_test = 8
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
MEAN, STD = [0.3904, 0.2603, 0.1758], [0.3059, 0.2112, 0.1446]
RE_MEAN = 0
DEVICE = torch.device("mps")
NR_EPOCHS = 20

def transform(image, mask, mean = None, std = None, image_dimensions = None):
    """INPUT: image, mask, with optional inputs: mean [3 channel tensor], standard deviation [3 channel tensor], and image dimensions[(H,W)]
    OUTPUT: image, mask
    DESC: Applies random horisontal flip, random rotation, and tensor transformation, as well as resizing (if image dimension given), and normalization (if mean and std given)
    """

    # RESIZE
    if image_dimensions:
        image = TF.resize(image, image_dimensions)
        mask = TF.resize(mask, image_dimensions)
    
    # RANDOMLY FLIP HORISONTALLY
    if np.random.randint(0,1) > 0.5: # Apply horisontal flip
        image = TF.hflip(image)
        mask = TF.hflip(mask)

    # RANDOM ROTATION
    angle = np.random.randint(-90, 90)
    image = TF.rotate(image, angle)
    mask = TF.rotate(mask, angle)

    # CONVERT TO TENSOR
    image = TF.to_tensor(image)
    mask = TF.to_tensor(mask) 
    if mean != None and STD != None:
        image = TF.normalize(image, mean, std)  # Use your mean and std values    
    mask = torch.where(mask > 0, torch.ones_like(mask), torch.zeros_like(mask))
    return image, mask

if RE_MEAN:
    dataset = ImageLoad(IMAGE_DATA_PATH, MASK_DATA_PATH, transform=transform, image_dimension=(IMAGE_HEIGHT, IMAGE_WIDTH))
    def get_mean_std(loader):
    # Variables to store sum and squared sum of all pixels
        channels_sum, channels_squared_sum, num_batches = 0, 0, 0

        for data, _ in loader:
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
        return mean, std

    # mean and std of entire data set
    # full_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    # MEAN, STD = get_mean_std(full_loader)

dataset = ImageLoad(IMAGE_DATA_PATH, MASK_DATA_PATH, transform=transform, mean=MEAN, std=STD, image_dimension=(IMAGE_HEIGHT, IMAGE_WIDTH))

# image_loader = data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True)
train_set, validation_set, test_set = data.random_split(dataset, [0.8, 0.1, 0.1])
train_loader = data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True)
test_loader = data.DataLoader(test_set, batch_size=batch_size_test, shuffle=False)
val_loader = data.DataLoader(validation_set, batch_size=batch_size_test, shuffle=False)

## VISUALIZE
def imshow(image, mask, preds = None):
    image_np = image.numpy()
    mask_np = mask.numpy()
    if preds != None:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        preds_np = preds.numpy()
        axs[2].imshow(np.transpose(preds_np, (1, 2, 0)), cmap='gray')
        axs[2].set_title('Predicted Mask')
        axs[2].axis('off')
    else:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(np.transpose(image_np, (1, 2, 0)))
    axs[0].set_title('Image')
    axs[0].axis('off')
    axs[1].imshow(np.transpose(mask_np, (1, 2, 0)), cmap='gray')
    axs[1].set_title('Mask')
    axs[1].axis('off')
    plt.show()

# Visualize an image and its mask
# dataiter = iter(train_loader)
# images, masks = next(dataiter) # Extracts the next batch of images
# imshow(images[0], masks[0])

## MODEL
MODEL = models.deeplabv3_resnet50(pretprogress=True, weights="DeepLabV3_ResNet50_Weights.DEFAULT") #DeepLab V3 model
MODEL.classifier[4] = torch.nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1)) # Changes last layer of DeepLab V3 model to give a binary output
OPTIMIZER = torch.optim.Adam(MODEL.parameters(), lr=0.001)
LOSS_FN = torch.nn.BCEWithLogitsLoss()
MODEL.to(DEVICE)

## TRAIN
def train(loader, model, optimizer, loss_fn): # trains one epoch
    loop = tqdm.tqdm(loader)
    
    for idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)["out"] # Predict
        loss = loss_fn(outputs, targets) # Measure loss

        optimizer.zero_grad() # Reset gradient
        loss.backward() #backwards propogation
        optimizer.step() # Update weights

        loop.set_postfix(loss = loss.item())

## EVALUATE
def evaluate(loader, model, device, loss_fn):
    loop = tqdm.tqdm(loader)
    
    model.eval()
    iou_score = 0
    sample_count = 0
    loss = 0

    with torch.no_grad():
        for idx, (data, targets) in enumerate(loop):
            sample_count += 1
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)["out"]
            preds = outputs > 0  # Assuming sigmoid activation
            loss += loss_fn(outputs, targets)
            iou_score += IoU(targets, preds)
    return loss/sample_count, iou_score/sample_count

def IoU(truth, prediction):
    truth_np = truth.cpu()
    prediction_np = prediction.cpu()
    truth_np = np.array(truth_np)
    prediction_np = np.array(prediction_np)
    intersection = np.logical_and(truth_np, prediction_np) #All cells with a 1-value
    union = np.logical_or(truth_np, prediction_np) #All cells where one of the masks are 1
    iou_score = np.sum(intersection) / np.sum(union) #IoU is calculated as the sum of intersections divided by the sum of union. I.e correlation divided by total predictions
    return iou_score # Best possible score is 1, worst is 0.

## RESULTS
def result(loader, model, device):
    # visualize image, ground truth mask and predicted mask
    dataiter = iter(loader)
    images, masks = next(dataiter) # Extracts the next batch of images
    images, masks = images.to(device), masks.to(device)
    output = model(images)["out"]
    pred = output > 0

    cpu_images = images.cpu()
    cpu_masks = masks.cpu()
    cpu_pred = pred.cpu()

    idx = np.random.randint(0,batch_size_test-1)
    imshow(cpu_images[idx], cpu_masks[idx], preds=cpu_pred[idx]) # show image and mask

def evaluation_plot(validation_loss, validation_iou):
    # Ensure the data is on the CPU and converted to a NumPy array
    for idx in range(NR_EPOCHS):
        if torch.is_tensor(validation_loss[idx]):
            validation_loss[idx] = validation_loss[idx].cpu().numpy()
        if torch.is_tensor(validation_iou[idx]):
            validation_iou[idx] = validation_iou[idx].cpu().numpy()
    x = [*range(NR_EPOCHS)]
    fig, axs = plt.subplots(1, 2, figsize=(10, 10))

    # Plot validation loss
    axs[0].plot(x, validation_loss)
    axs[0].set_title("Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    # Plot validation IoU
    axs[1].plot(x, validation_iou)
    axs[1].set_title("Validation IoU")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("IoU")
    plt.show()

def plot_validation_metrics(validation_loss, validation_iou, epochs):
    """
    Plots the validation loss and validation IoU for each epoch.

    Parameters:
    validation_loss (list): A list of validation loss values per epoch.
    validation_iou (list): A list of validation IoU values per epoch.
    NR_EPOCHS (int): The number of epochs.

    Returns:
    A matplotlib plot.
    """
    for idx in range(epochs):
        if torch.is_tensor(validation_loss[idx]):
            validation_loss[idx] = validation_loss[idx].cpu().numpy()
        if torch.is_tensor(validation_iou[idx]):
            validation_iou[idx] = validation_iou[idx].cpu().numpy()

    epochs = range(1, epochs + 1)

    fig, ax1 = plt.subplots()

    # Plot validation loss
    color = 'tab:red'
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Validation Loss', color=color)
    ax1.plot(epochs, validation_loss, color=color, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(epochs)  # Set x-ticks to only integers

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Validation IoU', color=color)
    ax2.plot(epochs, validation_iou, color=color, label='Validation IoU')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add a title and a grid
    plt.title('Validation Loss and IoU per Epoch')
    ax1.grid(True)

    # Add a legend
    fig.tight_layout()
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    plt.show()


## MAIN
#MODEL.to(DEVICE)
PATH = "state_dict_model"
TRAIN_MODEL = 0
VERBOSE = 1
best_val_loss, patience, validation_loss, validation_iou = np.Inf, 5 , np.zeros(NR_EPOCHS), np.zeros(NR_EPOCHS)

if TRAIN_MODEL:
    for epoch in range(NR_EPOCHS):
        train(train_loader, MODEL, OPTIMIZER, LOSS_FN)
        val_loss, val_iou = evaluate(val_loader, MODEL, DEVICE, LOSS_FN)
        validation_loss[epoch] = val_loss
        validation_iou[epoch] = val_iou
        print(f"EPOCH {epoch+1}\n",f"BCELoss: {val_loss}\n",f"Test IoU: {val_iou}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  # Reset counter
            # Save the model if it's the best so far
            torch.save(MODEL.state_dict(), PATH + f"_best_model.pt")
        else:
            patience_counter += 1
        if patience_counter > patience:
            print("Stopping early due to no improvement in validation loss")
            break
        # torch.save(MODEL.state_dict(), PATH + f"_epoch{epoch+1}")
    # Save validation metrics to CSV
    np.savetxt("validation_measures.csv", np.vstack((validation_loss, validation_iou)).T, delimiter=',', header='Validation Loss,Validation IoU', comments='', fmt='%f')
    if VERBOSE:
        plot_validation_metrics(validation_loss, validation_iou, NR_EPOCHS)
else:
    MODEL.load_state_dict(torch.load(PATH+ f"_best_model_cloud.pt"))
    test_loss, test_iou = evaluate(test_loader, MODEL, DEVICE, LOSS_FN)
    print(f"BCELoss: {test_loss}\n",f"Test IoU: {test_iou}")
result(test_loader, MODEL, DEVICE) # Just an example