#!/usr/bin/env python
# coding: utf-8

# #### SORU 4) Kendinizin oluşturacağı bir veri setinde K-means kümeleme algoritması ile farklı sayılarda alınan küme (K) değerleri için K=3, K=5 ve K=7 algoritmayı en az 5 kez çalıştırarak test setinizde başarıyı ölçünüz. Sisteminizin başarısını dikkate alarak algoritmanın zayıf yönlerini (küme merkezlerinin seçimi, iterasyon sayısı, kullanılan uzaklık metriği vs) yazınız. 

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# make_blobs ile 3 merkezli, 250 örneklik bir veri kümesi oluşturuyoruz.
X, y = make_blobs(n_samples=250, centers=3, random_state=42)

# oluşturduğumuz veri setini .csv uzantılı dosyaya çeviriyoruz.
df = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
df['target'] = y
df.to_csv('/Users/lenovo/Desktop/Soru_4.csv', index=False)

# K-means kümeleme algoritması oluşturuyoruz
def k_means(X, K, num_iterations=5):
    # rastgele K küme merkezi seçimi
    centers = X[np.random.choice(range(len(X)), size=K, replace=False)]
    for i in range(num_iterations):
        # uzaklık matrisini hesapla
        distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
        # en yakın merkeze göre örneklerin atanması
        labels = np.argmin(distances, axis=0)
        # yeni küme merkezlerinin hesaplanması
        centers = np.array([X[labels == j].mean(axis=0) for j in range(K)])
    return centers, labels

# Burada K-means algoritmasını test ediyoruz.
# K = 3, K = 5 ve K = 7 için algoritmayı en az 5 kez çalıştırıyoruz ve her bir durum için başarı oranını ölçüyoruz.
K_values = [3, 5, 7]
num_runs = 5
for K in K_values:
    print(f"K = {K} için:")
    accuracies = []
    for i in range(num_runs):
        centers, labels = k_means(X, K)
        accuracy = np.sum(labels == y) / len(y)
        accuracies.append(accuracy)
    print(f"Accuracies: {accuracies}")
    print(f"Mean accuracy: {np.mean(accuracies)}\n")


# In[ ]:




