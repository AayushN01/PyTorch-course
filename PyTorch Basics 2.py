#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

x = torch.tensor([5,3])
y = torch.tensor([2,1])

print(x*y)


# In[2]:


x = torch.zeros([2,5])
x


# In[3]:


x.shape


# In[4]:


y = torch.rand([2,5])
y


# In[6]:


y.view([1,10])


# In[ ]:




