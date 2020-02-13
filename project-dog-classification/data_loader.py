import os
from torchvision import datasets,transforms
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes
# Define a transform to normalize the data
data_dir = 'dogImages'
#train_dir = os.path.join('goodDogImages', 'train')
train_dir = os.path.join(data_dir, 'train')
validation_dir = os.path.join(data_dir, 'valid')
test_dir = os.path.join(data_dir, 'test')


# load and transform data using ImageFolder
image_resultion = 224

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transform = transforms.Compose([
    #transforms.Grayscale(),#let's if we manage in black and white
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.Resize(image_resultion), #keep propotion but put down a dimesion
    transforms.CenterCrop(image_resultion), #keep only the center square
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

train_data = ImageFolderWithPaths(train_dir, transform=data_transform)
classes = train_data.classes
test_data = ImageFolderWithPaths(test_dir, transform=data_transform)
test_classes = test_data.classes
validation_data = ImageFolderWithPaths(validation_dir, transform=data_transform)
validation_classes = validation_data.classes

# print out some data stats
print('Num training images: ', len(train_data))
print('Num test images: ', len(test_data))
print('Num validation images: ', len(validation_data))

# define dataloader parameters
batch_size = 20
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                        num_workers=num_workers, shuffle=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,num_workers=num_workers, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size,num_workers=num_workers, shuffle=True)

#train_loader,classes,test_loader,test_classes,validation_loader,validation_classes
num_classes = len(classes)
loaders_scratch = {'test':test_loader,'valid':validation_loader,'train':train_loader}


