#!/usr/bin/env python
# coding: utf-8

# # MACHINE LEARNING PROJECT

# ## Fotoğraflardaki El Yazısı Rakamları Otomatik Tanıma ve Anlamlandırma

# In[1]:


import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Bu işlem 1-2 dk sürebilir..
mnist = fetch_openml('mnist_784')


# In[2]:


mnist.data.shape



# In[3]:


def showimage(dframe, index):    
    some_digit = dframe.to_numpy()[index]
    some_digit_image = some_digit.reshape(28,28)

    plt.imshow(some_digit_image,cmap="binary")
    plt.axis("off")
    plt.show()


# In[8]:


showimage(mnist.data, 0)



# ### Split Data -> Training Set ve Test Set

# In[9]:



# test ve train oranı 1/7 ve 6/7
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, mnist.target, test_size=1/7.0, random_state=0)


# In[10]:


type(train_img)


# In[11]:


test_img_copy = test_img.copy()


# In[14]:


showimage(test_img_copy, 2)



# In[15]:


scaler = StandardScaler()

scaler.fit(train_img)

train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)


# 

# In[16]:


pca = PCA(.95)


# In[17]:


pca.fit(train_img)


# In[18]:


print(pca.n_components_)


# In[19]:


train_img = pca.transform(train_img)
test_img = pca.transform(test_img)



# In[20]:


logisticRegr = LogisticRegression(solver = 'lbfgs', max_iter=10000)



# In[21]:


logisticRegr.fit(train_img, train_lbl)



# In[23]:


logisticRegr.predict(test_img[0].reshape(1,-1))


# In[22]:


showimage(test_img_copy, 0)


# In[25]:


logisticRegr.predict(test_img[1].reshape(1,-1))


# In[24]:


showimage(test_img_copy, 1)


# In[26]:


showimage(test_img_copy, 9900)


# In[27]:


logisticRegr.predict(test_img[9900].reshape(1,-1))


# In[ ]:





# In[28]:


showimage(test_img_copy, 9999)


# In[29]:


logisticRegr.predict(test_img[9999].reshape(1,-1))


# In[ ]:





# In[ ]:






# In[30]:



logisticRegr.score(test_img, test_lbl)


# In[ ]:



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




