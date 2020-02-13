
# %%
import os
from torchvision import datasets,transforms
import torch

from data_loader import loaders_scratch,num_classes,train_loader,classes

# Visualize some sample data
import matplotlib.pyplot as plt
import numpy as np
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels,paths = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
plt.show()

# %% [markdown]
# %%

from my_dog_net import Net

# instantiate the CNN
model_scratch = Net(num_classes,'model_scratch.pt')
use_cuda = torch.cuda.is_available()
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
# %%
import torch.optim as optim
import torch.nn as nn
### TODO: select loss function
criterion_scratch = nn.CrossEntropyLoss()

### TODO: select optimizer
optimizer_scratch = optim.SGD(model_scratch.parameters(), lr=0.001)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


model_scratch.load()


def train(n_epochs, loaders, model, optimizer, criterion, use_cuda):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    last_saved_epoch = 0    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target,paths) in enumerate(loaders['train']):
            
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            ## find the loss and update the model parameters accordingly
            ## record the average training loss, using something like
            ## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            train_loss =train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
            loss.backward()
            optimizer.step()
            print('Epoch: {}/{} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tSaved Epoch: {}'.format(
                epoch, 
                batch_idx,
                train_loss,
                valid_loss_min,
                last_saved_epoch
            ))
            
        ######################    
        # validate the model #
        ######################
        model.eval()
        for batch_idx, (data, target,paths) in enumerate(loaders['valid']):
            
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss =valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} \tSaved Epoch: {}'.format(
            epoch, 
            train_loss,
            valid_loss_min,
            last_saved_epoch
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            if valid_loss_min != np.Inf:# first epoch after load we do not know if we got better.
                model.save()
                last_saved_epoch = epoch
            valid_loss_min = valid_loss
            
        if last_saved_epoch<(epoch-10):
            stop_now = object()
            #break
        if valid_loss<4.7:
            stop_now = object()

    # return trained model
    return model


# train the model
model_scratch = train(200, loaders_scratch, model_scratch, optimizer_scratch, 
                      criterion_scratch, use_cuda)



