## Standard Library
import os
import json

## External Libraries
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as functional
from torch.utils.data import Dataset, DataLoader
from skimage import io
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

## Batch Size
train_batch_size = 10
validation_batch_size = 10

## Learning Rate
learning_rate = 0.001

# Epochs (Consider setting high and implementing early stopping)
num_epochs = 200

# General Data Directory ##TODO: Please fill in the appropriate directory
data_dir = "./data"

## Segmentation + Colorization Paths
segmentation_data_dir = f"{data_dir}/segmentation/"
colorization_data_dir = f"{data_dir}/colorization/"

# Mask JSON
mask_json = f"{data_dir}/mapping.json"

## Image Transforms
tensor_transform = transforms.Compose([
        transforms.ToTensor(),
])

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
])

## Image Dataloader
class ImageDataset(Dataset):

    """
    ImageDataset
    """

    def __init__(self,
                 input_dir,
                 op,
                 mask_json_path,
                 transforms=None):
        """
        ##TODO: Add support for colorization dataset

        Args:
            input_dir (str): Path to either colorization or segmentation directory
            op (str): One of "train", "val", or "test" signifying the desired split
            mask_json_path (str): Path to mapping.json file
            transforms (list or None): Image transformations to apply upon loading.
        """
        self.transform = transforms
        self.op = op
        with open(mask_json_path, 'r') as f:
            self.mask = json.load(f)
        self.mask_num = len(self.mask)  # There are 6 categories: grey, dark grey, and black
        self.mask_value = [value for value in self.mask.values()]
        self.mask_value.sort()
        try:
            if self.op == 'train':
                self.data_dir = os.path.join(input_dir, 'train')
            elif self.op == 'val':
                self.data_dir = os.path.join(input_dir, 'validation')
            elif self.op == 'test':
                self.data_dir = os.path.join(input_dir, 'test')
            elif self.op == 'train_cor':
                self.data_dir = os.path.join(input_dir, 'train_cor')
            elif self.op == 'val_cor':
                self.data_dir = os.path.join(input_dir, 'validation_cor')
        except ValueError:
            print('op should be either train, val or test!')

    def __len__(self):
        """

        """
        return len(next(os.walk(self.data_dir))[1])

    def __getitem__(self,
                    idx):
        """

        """
        ## Load Image and Parse Properties
        if self.op == "train" or self.op == "test" or self.op == "val":
            img_name = str(idx) + '_input.jpg'
            mask_name = str(idx) + '_mask.png'
            img = io.imread(os.path.join(self.data_dir, str(idx), img_name))
            mask = io.imread(os.path.join(self.data_dir, str(idx), mask_name))
            if len(mask.shape) == 2:
                h, w  = mask.shape
            elif len(mask.shape) == 3:
                h, w, c = mask.shape
            ## Convert grey-scale label to one-hot encoding
            new_mask = np.zeros((h, w, self.mask_num))
            for idx in range(self.mask_num):
                #if the mask has 3 dimension use this code
                new_mask[:, :, idx] = mask[:,:,0] == self.mask_value[idx]
                #if the mask has 1 dimension use the code below
                #new_mask[:, :, idx] = mask == self.mask_value[idx]
            ## Transform image and mask


            if self.transform:
                img, mask = self.img_transform(img, new_mask)
            # ## Use dictionary to output
            # sample = {'img': img, 'mask': mask}
            # return sample
            return img, mask
        else:
            gray_img_name = str(idx) + '_gray.jpg'
            img_name = str(idx) + '_input.jpg'
            gray_img = io.imread(os.path.join(self.data_dir, str(idx), gray_img_name))
            img = io.imread(os.path.join(self.data_dir, str(idx), img_name))
            #gray_img = np.repeat(gray_img[np.newaxis, :, :], 3, axis=0)
            gray_img = np.stack((gray_img,) * 3, axis=-1)
            if self.transform:
                gray_img, img = self.img_transform(gray_img, img)
            return gray_img, img
        
    def img_transform(self,
                      img,
                      mask):
        """

        """
        ## Apply Transformations to Image and Mask
        img = self.transform(img)
        mask = self.transform(mask)
        return img, mask

## Functions for adding the convolution layer
def add_conv_stage(dim_in,
                   dim_out,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=True,
                   useBN=True):
    """

    """
    # Use batch normalization
    if useBN:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.BatchNorm2d(dim_out),
          nn.LeakyReLU(0.1)
        )
    # No batch normalization
    else:
        return nn.Sequential(
          nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU(),
          nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
          nn.ReLU()
        )

## Upsampling
def upsample(ch_coarse,
             ch_fine):
    """

    """
    return nn.Sequential(
                    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
                    nn.ReLU())


# U-Net
class UNET(nn.Module):

    """

    """
    def __init__(self, n_classes, useBN=True):
        """
        Args:
            n_classes (int): Number of classes
            useBN (bool): Turn Batch Norm on or off. (Hint: Using BatchNorm might help you achieve better performance.)
        """
        super(UNET, self).__init__()
        # Downgrade stages
        self.conv1 = add_conv_stage(3, 32, useBN=useBN)
        self.conv2 = add_conv_stage(32, 64, useBN=useBN)
        self.conv3 = add_conv_stage(64, 128, useBN=useBN)
        self.conv4 = add_conv_stage(128, 256, useBN=useBN)
        # Upgrade stages
        self.conv3m = add_conv_stage(256, 128, useBN=useBN)
        self.conv2m = add_conv_stage(128,  64, useBN=useBN)
        self.conv1m = add_conv_stage( 64,  32, useBN=useBN)
        # Maxpool
        self.max_pool = nn.MaxPool2d(2)
        # Upsample layers
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128,  64)
        self.upsample21 = upsample(64 ,  32)
        # weight initialization
        # You can have your own weight intialization. This is just an example.
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
        #TODO: Design your last layer & activations
        self.out_conv = nn.Conv2d(32, n_classes, kernel_size=1)


    def forward(self, x):
        """
        Forward pass
        """
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))

        conv4m_out_ = torch.cat((self.upsample43(conv4_out), conv3_out), 1)
        conv3m_out  = self.conv3m(conv4m_out_)

        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
        conv2m_out  = self.conv2m(conv3m_out_)

        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
        conv1m_out  = self.conv1m(conv2m_out_)

        #TODO: Design your last layer & activations

        out = self.out_conv(conv1m_out)
        out = torch.sigmoid(out)

        return out

##TODO: Finish implementing the multi-class DICE score function
def dice_score_image(prediction, target, n_classes):
    '''
      computer the mean dice score for a single image

      Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does
      Args:
          prediction (tensor): predictied labels of the image
          target (tensor): ground truth of the image
          n_classes (int): number of classes

      Returns:
          m_dice (float): Mean dice score over classes
    '''
    ## Should test image one by one
    assert prediction.shape[0] == 1 #This line can not be deleted
    ## TODO: Compute Dice Score for Each Class. Compute Mean Dice Score over Classes.
    dice_classes = np.zeros(n_classes)

    prediction_one_hot = functional.one_hot(prediction.squeeze(0).to(torch.int64), num_classes=n_classes)  # [256, 320, n_classes]
    prediction_one_hot = prediction_one_hot.permute(2, 0, 1).unsqueeze(0)  # [1, n_classes, 256, 320]

    for cl in range(n_classes):
        pred_flat = prediction_one_hot[:, cl].view(-1).float()
        target_flat = target[:, cl].view(-1).float()

        TP = (pred_flat * target_flat).sum()
        FP = (pred_flat * (1 - target_flat)).sum()
        FN = ((1 - pred_flat) * target_flat).sum()

        #When there is no ground truth of the class in this image
        #Give 1 dice score if False Positive pixel number is 0,
        #give 0 dice score if False Positive pixel number is not 0 (> 0).
        if target_flat.sum() == 0:
            dice_classes[cl] = 1 if FP == 0 else 0
        else:
            dice_classes[cl] = (2. * TP) / (2. * TP + FP + FN)
        
    return dice_classes.mean()


def dice_score_dataset(model, dataloader, num_classes, use_gpu=True):
    """
    Compute the mean dice score on a set of data.

    Note that multiclass dice score can be defined as the mean over classes of binary
    dice score. Dice score is computed per image. Mean dice score over the dataset is the dice
    score averaged across all images.

    Reminders: A false positive is a result that indicates a given condition exists, when it does not
               A false negative is a test result that indicates that a condition does not hold, while in fact it does

    Args:
        model (UNET class): Your trained model
        dataloader (DataLoader): Dataset for evaluation
        num_classes (int): Number of classes

    Returns:
        m_dice (float): Mean dice score over the input dataset
    """
    ## Number of Batches and Cache over Dataset
    n_batches = len(dataloader)
    scores = np.zeros(n_batches)
    ## Evaluate
    model.eval()
    idx = 0
    for data in dataloader:
        ## Format Data
        img, target = data
        if use_gpu:
            img = img.cuda()
            target = target.cuda()
        ## Make Predictions
        out = model(img)
        n_classes = out.shape[1]

        prediction = torch.argmax(out, dim = 1)
        scores[idx] = dice_score_image(prediction, target, n_classes)
        idx += 1
        img.cpu()
        target.cpu()

    ## Average Dice Score Over Images
    m_dice = scores.mean()
    return m_dice


## TODO: Implement DICE loss,
#  It should conform to to how we computer the dice score.
class DICELoss(nn.Module):
    def __init__(self, num_classes, eps=1e-5):
        super(DICELoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, prediction, target):
        dice_classes = torch.zeros(self.num_classes, device=prediction.device)

        for cl in range(self.num_classes):
            pred_cl = prediction[:, cl, ...].contiguous().view(-1)
            target_cl = target[:, cl, ...].contiguous().view(-1)

            inter = (pred_cl * target_cl).sum()
            union = pred_cl.sum() + target_cl.sum() + self.eps
            dice_classes[cl] = (2. * inter) / union

        dice_loss = 1 - dice_classes.mean()

        return dice_loss

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

"""
Actual UNet model running time
"""
train_dataset=ImageDataset(input_dir=segmentation_data_dir, op="train", mask_json_path=mask_json, transforms=img_transform)
validation_dataset=ImageDataset(input_dir=segmentation_data_dir, op="val", mask_json_path=mask_json, transforms=img_transform)
test_dataset=ImageDataset(input_dir=segmentation_data_dir, op="test", mask_json_path=mask_json, transforms=tensor_transform)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=validation_batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

## Initialize your unet
n_classes = 6
model_2 = UNET(n_classes)
model_2.to(device)
early_stopper = EarlyStopper(patience=5, min_delta=0)
num_epochs = 200
## Initialize Optimizer and Learning Rate Scheduler
optimizer = torch.optim.Adam(model_2.parameters(),lr=learning_rate)
criterion = DICELoss(num_classes=n_classes)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

test_dice_score = 0
trainLoss = []
valLoss = []
while test_dice_score < 0.6:
    trainLoss = []
    valLoss = []
    print("Start Training...")
    for epoch in range(num_epochs):
        ########################### Training #####################################
        print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
        # TODO: Design your own training section

        model_2.train()
        running_loss = 0.0

        for input, label in train_dataloader:
            input = input.cuda()
            label = label.cuda()

            optimizer.zero_grad()
            
            output = model_2(input)
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            input = input.cpu()
            label = label.cpu()
        scheduler.step()
        trainLoss.append(running_loss/len(train_dataloader))
            
        model_2.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input, label in validation_dataloader:
                input = input.cuda()
                label = label.cuda()

                output = model_2(input)
                val_loss += criterion(output, label).item()

                input = input.cpu()
                label = label.cpu()

        val_loss /= len(validation_dataloader)
        valLoss.append(val_loss)
        if early_stopper.early_stop(val_loss): 
            break

    with torch.no_grad():
        test_dice_score = dice_score_dataset(model_2, test_dataloader, n_classes)
        print(f"Test Dice Score: {test_dice_score}")

    epochs = list(range(1,len(trainLoss)+1))

"""
Finished running, printing out losses and plots
"""
for i in range(len(trainLoss)):
    print(f"Epoch {i+1}/{num_epochs} - Loss: {trainLoss[i]}")
    print(f"Epoch {i+1}/{num_epochs} - Validation Loss: {valLoss[i]}")

# Plotting
plt.figure(figsize=(10, 6))

# Plot the first set of accuracies
#for i in range(len(epochs)):
#    plt.plot(epochs[i], trainLoss_list[i], valLoss_list[i], color = 'r')1
plt.plot(trainLoss, color = 'blue')
plt.plot(valLoss, color = 'red')


# Setting title, labels, and other configurations
plt.title('train/validation losses over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'validation'])
plt.grid(True)

# Display the plot
plt.show()

# Free model from GPU Ram
model_2.to(torch.device("cpu"))