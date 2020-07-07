#!/usr/bin/env python
# coding: utf-8

# # KNN Algoritması Tutorial
# İmport Dataset
# 
# Dataset Tanımı
# 
# Dataset Görselleştirme
# 
# Knn algoritması açıklama
# 
# Knn with Sklearn
# 

# In[17]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[18]:


data=pd.read_csv("KNN(kanser).csv")
#id ve unnamed kısmı analizimiz için işe yaramadığı için kaldırıyoruz
data.drop(["id","Unnamed: 32"],axis=1,inplace=True)


# In[19]:


data.head()


# In[20]:


#malignant=M Kotu huylu tumor
#benign=B İyi huylu tumor

M=data[data.diagnosis=="M"]
B=data[data.diagnosis=="B"]


# In[21]:


#scatter plot
#tumor yarıcapına ve tumor dokusuna göre x y ekseninde analiz yapalım
plt.scatter(M.radius_mean,M.texture_mean,color="red",label="kotu",alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="iyi",alpha=0.3)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[22]:


#sınıflandırma yaparken m ve b değerleri(iyi huylu kötü huylu) yerine
#int değerler kullanıyoruz hata almamak için.0 ve 1 kullanacağız
data.diagnosis=[1 if each=="M" else 0 for each in data.diagnosis]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[23]:


#bu adımda normalizasyon yapacağız
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[24]:


#train test split
#verilerimizi eğitip test ediyoruz
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[25]:


#knn model
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) #dosya içindeki resimde yaptığımız örnek gibi
knn.fit(x_train,y_train)
predection=knn.predict(x_test)


# In[ ]:


#algoritmamız ne kadar doğru test skoruna bakalım
print(" {} nn score: {}".format(3,knn.score(x_test,y_test)))


# # n_neigbors değerini nasıl seçmeliyiz ?
# Biz burda bu değeri, 3 sectik ama max değeri bulmak için tek tek değerleri denemeliyiz
# 
# Bu işi kolaylaştırmak için aşagıdaki kodu inceliyelim

# In[28]:


#fin k value
score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test,y_test))
    
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("accuracy")
plt.show()
    


# Şekilde de görüldüğü üzere eğer k değeri=8 olunca
# 
# maximum verimi elde etmiş oluyoruz

# In[ ]:




