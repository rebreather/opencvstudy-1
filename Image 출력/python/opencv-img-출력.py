#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
print(cv2.__version__)


# In[5]:


import cv2

img_test = "test.jpg"

img = cv2.imread(img_test)

cv2.imshow('test img', img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# In[ ]:




