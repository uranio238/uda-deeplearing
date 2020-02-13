
# %%

# %% [markdown]
# %%
from data_loader import loaders_scratch,num_classes
from my_dog_net import Net
import torch

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net(num_classes,'model_scratch.pt')
use_cuda = torch.cuda.is_available()
# move tensors to GPU if CUDA is available
if use_cuda:
    model_scratch.cuda()
# %%
import numpy as np

model_scratch.load()
model_scratch.eval()

accuracy = []

for batch_idx, (data, target,paths) in enumerate(loaders_scratch['test']):
            
    # move to GPU
    if use_cuda:
        data, target = data.cuda(), target.cuda()
    
    output = model_scratch(data)
    output = output.argmax(1)
    score = torch.eq(target,output).sum()
    print(batch_idx,score)
    
    accuracy.append(score*1.0/target.shape[0])

percentage = sum(accuracy)/len(accuracy)




