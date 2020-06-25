#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torchvision


# In[2]:


import torch
import torchvision
from torchvision import transforms, datasets


# In[3]:


train = datasets.MNIST("", train=True, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))

test = datasets.MNIST("", train=False, download=True,
                       transform = transforms.Compose([transforms.ToTensor()]))


# In[7]:


trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle = True)


# In[8]:


trainset


# In[9]:


for data in trainset:
    print(data)
    break


# In[10]:


x, y = data[0][0], data[1][0]
print(y)


# In[11]:


print(x)


# In[19]:


import matplotlib.pyplot as plt


# In[20]:


#plt.imshow(data[0][0])


# In[21]:


print(data[0][0].shape)


# In[22]:


plt.imshow(data[0][0].view(28,28))
plt.show()


# In[24]:


print(data[0][0].shape)


# In[25]:


print(data[1][1].shape)


# In[34]:


total = 0
counter_dict = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}

for data in trainset:
    Xs, Ys = data
    for y in Ys:
        counter_dict[int(y)] += 1
        total += 1
print(counter_dict)


# In[35]:


for i in counter_dict:
    print(f"{i}: {counter_dict[i]/total*100}")


# In[ ]:




