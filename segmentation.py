##Segmentation part of the histology project##

##First we need to split the images into train, test, val
    # In this splitting, we are loading the images into the image-mask pairs

##Second, we need to transform the images from npy/jpeg/png format to tensors for the model to take as input##
    #We augment and transform (Noise, rotation, Intensity) to "make" more data to train on and make the model better at identifying and segmentation

##Third,we batch with data loader and feed into the network/pre-trained (U-net)

##Fourth, we train the model using U-net architechture. (From Scratch equivalent)
    # Define the loss function
    # What kind of Metrics to use?
    # Set-up training loop

##Evaluate performance
    # Metrics
    # inspecting one image

## we train the model using a more modern architechture - Seg ResNet
    # Define the loss function
    # What kind of Metrics to use?
    # Set-up training loop

##Evaluate performance
    # Metrics
    # inspecting one image

## Model Explainability??
####################################################################################################################################

                                                        ##Data Splitting##

####################################################################################################################################

import numpy as np
from sklearn.model_selection import train_test_split

# 1. Load data
images = np.load(r'.\data\Segmentation Project - Github\dataset\images.npy') 
masks = np.load(r'.\data\Segmentation Project - Github\dataset\masks.npy')
types = np.load(r'.\data\Segmentation Project - Github\Images\types.npy')

full_data_list = [
    {
        "image": images[i], 
        "label": masks[i], 
        "type": types[i], 
        "orig_index": i
    } 
    for i in range(len(images))
]

# Step 2: Now split the list of dictionaries
# train_test_split works on lists of dictionaries
train_files, temp_files = train_test_split(
    full_data_list, test_size=0.2, random_state=42
)

val_files, test_files = train_test_split(
    temp_files, test_size=0.5, random_state=42
)


###### FIX THE BACKGROUND layer to be from 6 -> 5. #####

# 1. Collapse the 6 channels into one channel of indices (0-5)
# This looks at the 6 channels for every pixel and picks the one with the '1'
semantic_masks = np.argmax(masks, axis=-1).astype(np.uint8)

# 2. Fix the Background (remap 6 to 5 if necessary)
# Based on your description, if Background is class 6, we move it to index 5
semantic_masks[semantic_masks == 6] = 5

class_names = [
    "0: Neoplastic (Cancer)", 
    "1: Inflammatory", 
    "2: Connective/Soft Tissue", 
    "3: Dead Cells", 
    "4: Epithelial", 
    "5: Background"
]

# 3. VERIFY: This MUST output [0 1 2 3 4 5]
print(f"Verified unique labels: {np.unique(semantic_masks)}")


####################################################################################################################################

                                                    ##Plotting Image and Mask##

####################################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

idx = 10
current_type = types[idx]
train_files[idx]["image"].shape # (256, 256, 3) 3 channels
train_files[idx]["label"].shape # (256, 256, 6) 6 channels
# 1. Access the arrays directly from dictionary
img_array = train_files[idx]["image"] #image
label_array = train_files[idx]["label"] #Mask

# 2. Pre-process for visualization
# Normalize image if it's 0-255 floats to avoid the clipping warning
if img_array.max() > 1.0:  # checking if the arrays are in the standard 0–255 range (integers/floats) or already normalized (0–1)
    img_array = img_array / 255.0 #matplotlib wants float arrays between 0 to 1. 

# Collapse 6-channel One-Hot mask into a single 2D class map (0 to 5)
if label_array.ndim == 3 and label_array.shape[-1] == 6:  #check if mask is 3D array and has 6 channels
    label_viz = np.argmax(label_array, axis=-1)           #For every pixel:
                                                            # axis=-1 looks at the last dimension of label_array. It will look at all 6 channels one by one.
                                                            # np.arg.max returns the index/postition of the highest value
else:
    label_viz = label_array

# 3. Plotting

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(img_array)
plt.title(f"Image (Tissue: {current_type})")
plt.axis('off')

plt.subplot(1, 2, 2)
# Using 'jet' or 'viridis' helps distinguish the 6 different classes
plt.imshow(label_viz, cmap='jet_r')  #cmap is colourmap
plt.title(f"Multi-class Mask")
plt.axis('off')

# Add Legend to the Right
# Get the colors from the 'jet_r' colormap
cmap = plt.get_cmap('jet_r')
# np.linspace helps us pick 6 colors evenly from the map
colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

# Create the colored patches
patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]

# Place the legend
# bbox_to_anchor(1.05, 1) moves it to the right of the current subplot
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.tight_layout()
plt.show()    

####################################################################################################################################

                                                        ##Data Loading##

####################################################################################################################################

import numpy as np
from torch.utils.data import DataLoader

from monai.data import Dataset
from monai.transforms import (
    Compose, EnsureChannelFirstd, EnsureTyped,
    ScaleIntensityd, SpatialPadd, RandFlipd, RandRotate90d,
    RandAffined, RandAdjustContrastd, RandGaussianNoised,
    Activations, AsDiscrete, AsDiscreted
)

# Define the region of interest (ROI) size for cropping or patch-based processing.
roi_size = (256, 256)

# Define the preprocessing and augmentation pipeline for training images and labels.
train_transforms = Compose([
    #LoadImaged(keys=["image", "label"]),           # Fixed - don't need this as the images are already loaded in RAM. If this stays, MONAI will try to find a file path
    EnsureChannelFirstd(keys=["image", "label"],channel_dim = -1),  # Fixed, channel_dim = -1 is so that the last dimension is the channels.
    EnsureTyped(keys=["image", "label"]),          # Fixed

    ScaleIntensityd(keys=["image"]),                               # Fixed
    SpatialPadd(keys=["image", "label"], spatial_size=roi_size),   # Fixed
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),  # Augmentation
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),  # Augmentation
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),     # Augmentation
    RandAffined(keys=["image", "label"], prob=0.25,                # Augmentation
                rotate_range=np.deg2rad(10.0),
                translate_range=(8, 8),
                scale_range=(0.1, 0.1),
                padding_mode="zeros",
                mode=("bilinear","nearest")),
    RandAdjustContrastd(keys=["image"], prob=0.3, gamma=(0.7,1.5)),    # Augmentation
    RandGaussianNoised(keys=["image"], prob=0.15, mean=0.0, std=0.01), # Augmentation
    AsDiscreted(keys=["label"], threshold=0.5),                        # Fixed (label)
])

# Define the transformation pipeline for validation/test data (no random augmentations).
val_test_transforms = Compose([
    #LoadImaged(keys=["image", "label"]),                                # Fixe- don't need this as the images are already loaded in RAM. If this stays, MONAI will try to find a file path
    EnsureChannelFirstd(keys=["image", "label"],channel_dim = -1),      # Fixed
    EnsureTyped(keys=["image", "label"]),                               # Fixed
    ScaleIntensityd(keys=["image"]),                                    # Fixed
    SpatialPadd(keys=["image", "label"], spatial_size=roi_size),        # Fixed
    AsDiscreted(keys=["label"], threshold=0.5),                         # Fixed (label)
])

# Load the training, validation and test datasets, applying the transformations and caching all data in memory for fast access.
train_ds = Dataset(train_files, transform=train_transforms)
val_ds = Dataset(val_files, transform=val_test_transforms)  
test_ds = Dataset(test_files, transform=val_test_transforms)

####################################################################################################################################

                                                        ##Data Batching##

####################################################################################################################################

from torch.utils.data import DataLoader


# Set-up batch size
BATCH_SIZE = 16

# 'masks' is (2722, 256, 256, 6) array
# We MUST convert this back to (2722, 256, 256) integers 0-5 
# for the standard MONAI pipeline to work correctly.
if masks.shape[-1] == 6:
    print("Converting one-hot masks to integer labels...")
    masks_int = np.argmax(masks, axis=-1).astype(np.uint8)
else:
    masks_int = masks.astype(np.uint8)

# Now check for the rogue '6' in the README
masks_int[masks_int == 6] = 5 

print(f"Final Unique Labels: {np.unique(masks_int)}") # MUST be [0 1 2 3 4 5]


# Prepare PyTorch DataLoader objects for batch processing
# Shuffle the training data for better training performance
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Load one batch to inspect
train_batch= next(iter(train_loader))

print("Train batch shape:", train_batch["image"].shape) #Train batch shape: torch.Size([16, 3, 512, 512])
print("Label batch shape:", train_batch["label"].shape) #Label batch shape: torch.Size([16, 6, 512, 512])


####################################################################################################################################

                                                    ##Display Image-Mask-Overlay##

####################################################################################################################################

from monai.data import decollate_batch
import matplotlib.pyplot as plt
import torch

# Define function to show image/mask
def show_image_label(batch, index):

    items = decollate_batch(batch)       # list of {"image": (C,H,W), "label": (1,H,W)}
    sample = items[index]                # {"image": (C,H,W), "label": (1,H,W)}

    img  = sample["image"].clamp(0,1)          # (C,H,W) tensor #Sometimes, data augmentations like RandGaussianNoised or RandAdjustContrastd can push pixel values slightly below 0 or above 1.
    mask = torch.argmax(sample["label"], dim=0) # (6,H,W) tensor

    img_hwc  = img.permute(1,2,0).cpu().numpy() # (C,H,W) -> (H,W,C)
    mask_hw  = mask.squeeze(0).cpu().numpy()    # (1,H,W) -> (H,W)

    plt.figure(figsize=(18,6))
    # image
    plt.subplot(1,3,1)
    plt.imshow(img_hwc)
    plt.axis("off")
    plt.title(f"Image (Tissue: {current_type})")

    # mask
    plt.subplot(1,3,2)
    plt.imshow(mask_hw)
    plt.axis("off")
    plt.title("mask")

    # overlay
    plt.subplot(1,3,3)
    plt.imshow(img_hwc)
    plt.imshow(mask_hw, alpha=0.35, cmap="jet_r", interpolation="nearest")
    plt.axis("off")
    plt.title("overlay")

    #Legend
    # Get the colors from the 'jet_r' colormap
    cmap = plt.get_cmap('jet_r')
    # np.linspace helps us pick 6 colors evenly from the map
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]

    # Create the colored patches
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]

    # Place the legend
    # bbox_to_anchor(1.05, 1) moves it to the right of the current subplot
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='lower left', borderaxespad=0.)

    plt.tight_layout()
    plt.show()   

# Show image/mask from batch

show_image_label(train_batch, 4)

####################################################################################################################################

                                                        ##U-Net##

####################################################################################################################################

import torch
import torch.nn as nn

# Buildig block
class DoubleConv(nn.Module):
    """(Conv3x3 -> BN -> ReLU) x 2"""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

# Basic U-Net (2D) with transposed conv upsampling
class UNet2D(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=16):
        super().__init__()
        # Encoder
        self.enc1 = DoubleConv(in_channels,       base_ch)       # -> C=64
        self.pool1 = nn.MaxPool2d(2)                             # /2
        self.enc2 = DoubleConv(base_ch,           base_ch*2)     # -> 128
        self.pool2 = nn.MaxPool2d(2)                             # /2
        self.enc3 = DoubleConv(base_ch*2,         base_ch*4)     # -> 256
        self.pool3 = nn.MaxPool2d(2)                             # /2
        self.enc4 = DoubleConv(base_ch*4,         base_ch*8)     # -> 512
        self.pool4 = nn.MaxPool2d(2)                             # /2

        # Bottleneck
        self.bottleneck = DoubleConv(base_ch*8,   base_ch*16)    # -> 1024

        # Decoder (transposed conv upsampling) + explicit skip concatenations
        self.up4  = nn.ConvTranspose2d(base_ch*16, base_ch*8,  kernel_size=2, stride=2)
        self.dec4 = DoubleConv(base_ch*16,         base_ch*8)   # cat(up4, enc4)

        self.up3  = nn.ConvTranspose2d(base_ch*8,  base_ch*4,  kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_ch*8,          base_ch*4)   # cat(up3, enc3)

        self.up2  = nn.ConvTranspose2d(base_ch*4,  base_ch*2,  kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_ch*4,          base_ch*2)   # cat(up2, enc2)

        self.up1  = nn.ConvTranspose2d(base_ch*2,  base_ch,    kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_ch*2,          base_ch)     # cat(up1, enc1)

        # 1x1 conv -> class logits
        self.head = nn.Conv2d(base_ch, num_classes, kernel_size=1)

    def forward(self, x, return_shapes=False):
        # ----- Encoder -----
        x1 = self.enc1(x)             # skip 1
        x2 = self.enc2(self.pool1(x1))# skip 2
        x3 = self.enc3(self.pool2(x2))# skip 3
        x4 = self.enc4(self.pool3(x3))# skip 4

        # ----- Bottleneck -----
        xb = self.bottleneck(self.pool4(x4))

        # ----- Decoder with explicit skips -----
        u4 = self.up4(xb)                         # upsample
        d4 = self.dec4(torch.cat([u4, x4], dim=1))# skip from enc4

        u3 = self.up3(d4)
        d3 = self.dec3(torch.cat([u3, x3], dim=1))# skip from enc3

        u2 = self.up2(d3)
        d2 = self.dec2(torch.cat([u2, x2], dim=1))# skip from enc2

        u1 = self.up1(d2)
        d1 = self.dec1(torch.cat([u1, x1], dim=1))# skip from enc1

        out = self.head(d1)  # (B, num_classes, H, W)

        return out
    
model = UNet2D(in_channels=3, num_classes=6, base_ch=32) #in_channels=3 to match 3 channels of Image array
                                                         #By setting num_classes=6, this final $1 \times 1$ convolution will take the high-level features and compress them into 6 separate maps.

print(model)

####################################################################################################################################

                                                        ##U-Net -Training##

####################################################################################################################################

import random
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Define Training Variables and Compute Utilization                           #
# --------------------------------------------------------------------------- #
# Set the device for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# For reproducibility (weight init, etc.)
SEED = 54
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define training variables
NUM_EPOCHS = 30
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4

# Output file path for saving the best model weights
CHECKPOINT_FILE = "checkpoint_simpleunet.pt"

# Move the model to the specified device (CPU or GPU)
model = model.to(DEVICE)

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

# --------------------------------------------------------------------------- #
# Define Loss, Optimizer and Performance Metrics                              #
# --------------------------------------------------------------------------- #
# Define the loss function as DiceLoss.
criterion = DiceCELoss(sigmoid=True)

# Use the AdamW optimizer to update model parameters
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)

# Set up metrics
train_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
val_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Set up scheduler
scheduler = ReduceLROnPlateau(optimizer)

from tqdm import tqdm

# Set-up reporting interval
REPORT_INTERVAL = 5  # print losses every 5 epochs

# Set-up checkpointing and early stopping
best_val_loss = float("inf")
no_improve_count = 0
patience = 5
min_delta = 1e-4   # minimum change to qualify as improvement

# Set-up tracking of loss/metric per epoch
running_train_loss = []
running_train_metric = []
running_val_loss = []
running_val_metric = []

# Epoch loop
for epoch in range(NUM_EPOCHS):

    # ---- Training ----
    model.train()

    train_loss_sum = 0.0
    train_metric.reset()

    for batch in tqdm(train_loader): # show progress bar with tqdm
        xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

        # Forward + loss
        output = model(xb)
        loss = criterion(output, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss + metric
        train_loss_sum += loss.item()
        train_metric(post_pred(output), yb)

    # Calculate average training loss/metric for epoch
    avg_train_loss = train_loss_sum / max(1, len(train_loader))
    avg_train_metric = train_metric.aggregate().item()

    # Track training loss/metric per epoch
    running_train_loss.append(avg_train_loss)
    running_train_metric.append(avg_train_metric)

    # ---- Validation ----
    model.eval()
    val_loss_sum = 0.0
    val_metric.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader): # show progress bar with tqdm
            xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            val_loss_sum += loss.item()
            val_metric(post_pred(output), yb)

    # Calculate average validation loss/metric for epoch
    avg_val_loss = val_loss_sum / max(1, len(val_loader))
    avg_val_metric = val_metric.aggregate().item()

    # Track validation loss/metric per epoch
    running_val_loss.append(avg_val_loss)
    running_val_metric.append(avg_val_metric)

    # ---- Scheduler step (after validation) ----
    scheduler.step(avg_val_loss)

    # ---- Checkpointing + Early Stopping ----
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        no_improve_count = 0

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            CHECKPOINT_FILE
        )
        print(f"Epoch {epoch+1}: Saved new best model (loss={best_val_loss:.4f})")
    else:
        no_improve_count += 1
        print(f"Epoch {epoch+1}: No improvement ({no_improve_count}/{patience})")

        if no_improve_count >= patience:
            print(f"Epoch {epoch+1}: Early stopping triggered.")
            break

    # ---- Report ----
    if (epoch + 1) == 1 or (epoch + 1) % REPORT_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss {avg_train_loss:.4f}  DICE {avg_train_metric:.3f} | "
            f"Val loss {avg_val_loss:.4f} DICE {avg_val_metric:.3f} | "
        )


####################################################################################################################################

                                                        ##U-Net -Visualisations##

####################################################################################################################################


# Visualization loss curve                                                #
plt.figure(figsize=(8,5))
plt.plot(running_train_loss, label='Train Loss')
plt.plot(running_val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curves for U-NET')
plt.legend()
plt.grid(True)
plt.show()

# Visualization metric curves                                                #
plt.figure(figsize=(8,5))
plt.plot(running_train_metric, label='Train Metric')
plt.plot(running_val_metric, label='Validation Metric')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training & Validation Metric Curves for U-NET')
plt.legend()
plt.grid(True)
plt.show()

# Restore best model for evaluation
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

# Set up metric
test_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Compute Dice across the entire test set using the existing test_loader
model.eval()
with torch.no_grad():
    for batch in test_loader:
        xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        out = model(xb)

        test_metric(post_pred(out), yb)

test_dice = test_metric.aggregate().item()
print(f"Test Dice Coeff: {test_dice:.4f}") #Test Dice Coeff: 0.7012


####################################################################################################################################

                                                    ##Look at predictions for 1 batch##

####################################################################################################################################

import torch, matplotlib.pyplot as plt
import torch, matplotlib.pyplot as plt

# Get predictions for 1 batch
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))                # xb: (B,C,H,W), yb: (B,1,H,W) or (B,H,W)
    xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
    output = model(xb)                              # (B,1,H,W) for binary

# This creates a (B, 6, H, W) tensor where each channel is a probability
pred = torch.softmax(output, dim=1)

# Define function to show image, true mask, predicted mask
def show_prediction(image, yb, pred, idx,class_names):
    #Tissue type
    tissue_name = batch["type"][idx]
    
    # Image
    img = image[idx].clamp(0,1)                            # (C,H,W)
    img2show = img.permute(1,2,0).detach().cpu().numpy()  # (H,W,C)

    # True Labels 
    # label_dict["label"] is (B, 6, H, W)
    true_hw = torch.argmax(yb[idx], dim=0).detach().cpu().numpy()
    
    #Predicted Labels
    # pred_dict["label"] is (B, 6, H, W)
    pred_hw = torch.argmax(pred[idx], dim=0).detach().cpu().numpy()

    plt.figure(figsize=(18, 5))
    
    # 2. Define Subplots
    plt.subplot(1, 3, 1)
    plt.imshow(img2show)
    plt.axis("off") 
    plt.title(f"Image | Tissue: {tissue_name}")

    plt.subplot(1, 3, 2)
    plt.imshow(true_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Model Prediction")

    # 3. Create Legend
    cmap = plt.get_cmap('jet_r')
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]

    # Place legend
    ax = plt.subplot(1, 3, 2)
    ax.legend(handles=patches, bbox_to_anchor=(0.5, -0.15), 
               loc='upper center', ncol=3, frameon=False)

    # 4. Final Layout Clean-up
    plt.show()



# Plot image, true mask, predicted mask
for idx in range(24):
    show_prediction(xb, yb, pred, idx, class_names=class_names)


#show_prediction(xb, batch, pred, 5)

####################################################################################################################################

                                                        ##SegResNet -Training##

####################################################################################################################################
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Garbage collect + free CUDA cache
import gc, torch
gc.collect()
torch.cuda.empty_cache()

from monai.networks.nets import SegResNet

model = SegResNet(
    spatial_dims=2,        # 2D
    in_channels=3,         # RGB
    out_channels=6,        
    init_filters=32,
    dropout_prob=0.2
)


import random
import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Define Training Variables and Compute Utilization                           #
# --------------------------------------------------------------------------- #
# Set the device for computation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", DEVICE)

# For reproducibility (weight init, etc.)
SEED = 54
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Define training variables
NUM_EPOCHS = 45
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Output file path for saving the best model weights
CHECKPOINT_FILE = "checkpoint_monai_segresnet.pt"

# Move the model to the specified device (CPU or GPU)
model = model.to(DEVICE)

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from monai.metrics import DiceMetric
from monai.losses import DiceCELoss

# --------------------------------------------------------------------------- #
# Define Loss, Optimizer and Performance Metrics                              #
# --------------------------------------------------------------------------- #
weights = torch.tensor([8.0, 2.0, 2.0, 2.0, 2.0, 1.0]).to(DEVICE)

criterion = DiceCELoss(
    softmax=True, 
    to_onehot_y=False, 
    weight=weights  # Apply weights to the CrossEntropy part of the loss
)

# Use the AdamW optimizer to update model parameters
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=LEARNING_RATE,
                              weight_decay=WEIGHT_DECAY)

# Set up metrics
train_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
val_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)

post_pred = Compose([
    Activations(softmax=True), 
    AsDiscrete(argmax=True, to_onehot=6, dim =1)

]) #Softmax is for multi class 
# Set up scheduler
scheduler = ReduceLROnPlateau(optimizer)

####################################################################################################################################

                                                   ##SegResNet -Training Loop##

####################################################################################################################################
from tqdm import tqdm

# Set-up reporting interval
REPORT_INTERVAL = 5  # print losses every 5 epochs

# Set-up checkpointing and early stopping
best_val_loss = float("inf")
no_improve_count = 0
patience = 5
min_delta = 1e-4   # minimum change to qualify as improvement

# Set-up tracking of loss/metric per epoch
running_train_loss = []
running_train_metric = []
running_val_loss = []
running_val_metric = []

# Epoch loop
for epoch in range(NUM_EPOCHS):

    # ---- Training ----
    model.train()

    train_loss_sum = 0.0
    train_metric.reset()


    for batch in tqdm(train_loader): # show progress bar with tqdm
        xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)

        # Forward + loss
        output = model(xb)
        loss = criterion(output, yb)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate training loss + metric
        train_loss_sum += loss.item()
        train_metric(post_pred(output), yb)

    # Calculate average training loss/metric for epoch
    avg_train_loss = train_loss_sum / max(1, len(train_loader))
    avg_train_metric = train_metric.aggregate().item()

    # Track training loss/metric per epoch
    running_train_loss.append(avg_train_loss)
    running_train_metric.append(avg_train_metric)

    # ---- Validation ----
    model.eval()
    val_loss_sum = 0.0
    val_metric.reset()

    with torch.no_grad():
        for batch in tqdm(val_loader): # show progress bar with tqdm
            xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
            output = model(xb)
            loss = criterion(output, yb)
            val_loss_sum += loss.item()
            val_metric(post_pred(output), yb)

    # Calculate average validation loss/metric for epoch
    avg_val_loss = val_loss_sum / max(1, len(val_loader))
    avg_val_metric = val_metric.aggregate().item()

    # Track validation loss/metric per epoch
    running_val_loss.append(avg_val_loss)
    running_val_metric.append(avg_val_metric)

    # Scheduler step (after validation)
    scheduler.step(avg_val_loss)

    # ---- Checkpointing + Early Stopping ----
    if avg_val_loss < best_val_loss - min_delta:
        best_val_loss = avg_val_loss
        no_improve_count = 0

        # Save checkpoint
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            },
            CHECKPOINT_FILE
        )
        print(f"Epoch {epoch+1}: Saved new best model (loss={best_val_loss:.4f})")
    else:
        no_improve_count += 1
        print(f"Epoch {epoch+1}: No improvement ({no_improve_count}/{patience})")

        if no_improve_count >= patience:
            print(f"Epoch {epoch+1}: Early stopping triggered.")
            break

    # ---- Report ----
    if (epoch + 1) == 1 or (epoch + 1) % REPORT_INTERVAL == 0 or (epoch + 1) == NUM_EPOCHS:
        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train loss {avg_train_loss:.4f} DICE {avg_train_metric:.3f} | "
            f"Val loss {avg_val_loss:.4f} DICE {avg_val_metric:.3f} | "
        )






####################################################################################################################################

                                                    ##SegResNet -Visualisations##

####################################################################################################################################

import matplotlib.pyplot as plt

# Visualization loss curve                                                #
plt.figure(figsize=(8,5))
plt.plot(running_train_loss, label='Train Loss')
plt.plot(running_val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss Curves')
plt.legend()
plt.grid(True)
plt.show()

# Visualization metric curves                                                #
plt.figure(figsize=(8,5))
plt.plot(running_train_metric, label='Train Metric')
plt.plot(running_val_metric, label='Validation Metric')
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Training & Validation Metric Curves')
plt.legend()
plt.grid(True)
plt.show()

# Restore best model for evaluation
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])

# Set up metric
test_metric = DiceMetric(include_background=True, reduction="mean", ignore_empty=True)
post_pred = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])

# Compute Dice across the entire TEST set using the existing test_loader
model.eval()
with torch.no_grad():
    for batch in test_loader:
        xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
        out = model(xb)

        test_metric(post_pred(out), yb)

test_dice = test_metric.aggregate().item()
print(f"Test Dice Coeff: {test_dice:.4f}")

# Aggregate the metric for each class separately
class_dice = test_metric.aggregate(reduction="mean_batch")

for i, name in enumerate(class_names):
    print(f"Class {i} ({name}) Dice: {class_dice[i].item():.4f}")

##############For DiceCELoss#####################
# print(f"Test Dice Coeff: {test_dice:.4f}")
#Test Dice Coeff: 0.6789

#Class 0 (0: Neoplastic (Cancer)) Dice: 0.6433
#Class 1 (1: Inflammatory) Dice: 0.4955
#Class 2 (2: Connective/Soft Tissue) Dice: 0.4593
#Class 3 (3: Dead Cells) Dice: 0.0000
#Class 4 (4: Epithelial) Dice: 0.6189
#Class 5 (5: Background) Dice: 0.9444

##############For DiceFocalLoss#####################
#print(f"Test Dice Coeff: {test_dice:.4f}")
#Test Dice Coeff: 0.6296
#Class 0 (0: Neoplastic (Cancer)) Dice: 0.6431
#Class 1 (1: Inflammatory) Dice: 0.2859
#Class 2 (2: Connective/Soft Tissue) Dice: 0.4772
#Class 3 (3: Dead Cells) Dice: 0.0000
#Class 4 (4: Epithelial) Dice: 0.3805
#Class 5 (5: Background) Dice: 0.9363

##############For 16 init_filters and 45 epochs#####################
#print(f"Test Dice Coeff: {test_dice:.4f}")
#Test Dice Coeff: 0.6782
# Aggregate the metric for each class separately
#
#Class 0 (0: Neoplastic (Cancer)) Dice: 0.5840
#Class 1 (1: Inflammatory) Dice: 0.5292
#Class 2 (2: Connective/Soft Tissue) Dice: 0.4729
#Class 3 (3: Dead Cells) Dice: 0.0000
#Class 4 (4: Epithelial) Dice: 0.6239
#Class 5 (5: Background) Dice: 0.9550

##############For 32 init_filters and 30 epochs#####################
#>>> print(f"Test Dice Coeff: {test_dice:.4f}")
#Test Dice Coeff: 0.6718
# Aggregate the metric for each class separately

#Class 0 (0: Neoplastic (Cancer)) Dice: 0.6012
#Class 1 (1: Inflammatory) Dice: 0.5070
#Class 2 (2: Connective/Soft Tissue) Dice: 0.4844
#Class 3 (3: Dead Cells) Dice: 0.0000
#Class 4 (4: Epithelial) Dice: 0.5234
#Class 5 (5: Background) Dice: 0.9571


##############For 32 init_filters and 45 epochs and cancer weight = 8.0#####################
#print(f"Test Dice Coeff: {test_dice:.4f}")
#Test Dice Coeff: 0.6683
# Aggregate the metric for each class separately

#Class 0 (0: Neoplastic (Cancer)) Dice: 0.5659
#Class 1 (1: Inflammatory) Dice: 0.5136
#Class 2 (2: Connective/Soft Tissue) Dice: 0.4707
#Class 3 (3: Dead Cells) Dice: 0.0000
#Class 4 (4: Epithelial) Dice: 0.6008
#Class 5 (5: Background) Dice: 0.9527

####################################################################################################################################

                                                    ##Look at predictions for 1 batch##

####################################################################################################################################
import torch, matplotlib.pyplot as plt

# Get predictions for 1 batch
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))                # xb: (B,C,H,W), yb: (B,1,H,W) or (B,H,W)
    xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
    output = model(xb)                              # (B,1,H,W) for binary

# 1. Convert raw scores to probabilities using Softmax
probs = torch.softmax(output, dim=1) 

# 2. Pick the winner (Argmax) 
# This gives you a map of integers (0-5)
pred_class_map = torch.argmax(probs, dim=1)
converter = AsDiscrete(to_onehot=6)
pred = torch.stack([converter(p) for p in pred_class_map])


# Define function to show image, true mask, predicted mask
def show_prediction(image, yb, pred, idx,class_names):
    #Tissue type
    tissue_name = batch["type"][idx]
    
    # Image
    img = image[idx].clamp(0,1)                            # (C,H,W)
    img2show = img.permute(1,2,0).detach().cpu().numpy()  # (H,W,C)

    # True Labels 
    # label_dict["label"] is (B, 6, H, W)
    true_hw = torch.argmax(yb[idx], dim=0).detach().cpu().numpy()
    
    #Predicted Labels
    # pred_dict["label"] is (B, 6, H, W)
    pred_hw = torch.argmax(pred[idx], dim=0).detach().cpu().numpy()

    plt.figure(figsize=(18, 5))
    
    # 2. Define Subplots
    plt.subplot(1, 3, 1)
    plt.imshow(img2show)
    plt.axis("off") 
    plt.title(f"Image | Tissue: {tissue_name}")

    plt.subplot(1, 3, 2)
    plt.imshow(true_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Model Prediction")

    # 3. Create Legend
    cmap = plt.get_cmap('jet_r')
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]

    # Place legend
    ax = plt.subplot(1, 3, 2)
    ax.legend(handles=patches, bbox_to_anchor=(0.5, -0.15), 
               loc='upper center', ncol=3, frameon=False)

    # 4. Final Layout Clean-up
    plt.show()



# Plot image, true mask, predicted mask
for idx in range(16):
    show_prediction(xb, yb, pred, idx, class_names=class_names)


import torch, matplotlib.pyplot as plt
import torch, matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Get predictions for 1 batch
model.eval()
with torch.no_grad():
    batch = next(iter(test_loader))                # xb: (B,C,H,W), yb: (B,1,H,W) or (B,H,W)
    xb, yb = batch["image"].to(DEVICE), batch["label"].to(DEVICE)
    output = model(xb)                              # (B,1,H,W) for binary

# This creates a (B, 6, H, W) tensor where each channel is a probability
pred = torch.softmax(output, dim=1)

# Define function to show image, true mask, predicted mask
def show_prediction(image, yb, pred, idx,class_names):
    #Tissue type
    tissue_name = batch["type"][idx]
    
    # Image
    img = image[idx].clamp(0,1)                            # (C,H,W)
    img2show = img.permute(1,2,0).detach().cpu().numpy()  # (H,W,C)

    # True Labels 
    # label_dict["label"] is (B, 6, H, W)
    true_hw = torch.argmax(yb[idx], dim=0).detach().cpu().numpy()
    
    #Predicted Labels
    # pred_dict["label"] is (B, 6, H, W)
    pred_hw = torch.argmax(pred[idx], dim=0).detach().cpu().numpy()

    plt.figure(figsize=(18, 5))
    
    # 2. Define Subplots
    plt.subplot(1, 3, 1)
    plt.imshow(img2show)
    plt.axis("off") 
    plt.title(f"Image | Tissue: {tissue_name}")

    plt.subplot(1, 3, 2)
    plt.imshow(true_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_hw, cmap='jet_r')
    plt.axis("off")
    plt.title("Model Prediction")

    # 3. Create Legend
    cmap = plt.get_cmap('jet_r')
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_names))]
    patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]

    # Place legend
    ax = plt.subplot(1, 3, 2)
    ax.legend(handles=patches, bbox_to_anchor=(0.5, -0.15), 
               loc='upper center', ncol=3, frameon=False)

    # 4. Final Layout Clean-up
    plt.show()



# Plot image, true mask, predicted mask
for idx in range(24):
    show_prediction(xb, yb, pred, idx, class_names=class_names)