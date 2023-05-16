#!/usr/bin/env python
# coding: utf-8

# ### SORU 3) 
# En az 10 özellik ve bu özelliklerden en az 3 tanesinin sürekli (continuous) olması koşuluyla , 3 sınıf ve 400 örneğin bulunduğu bir veri seti belirleyiniz. 

# - Oluşturulan veri setindeki üç özellik **(x_coordinate, y_coordinate ve z_coordinate)** süreklidir.
# - Diğer 7 özellik **(brightness, distance, spectral_intensity, texture_1, texture_2, texture_3 ve texture_4)** ise sayısal (numerical) ancak sürekli değildir.
# - Ayrıca, veriler **3 sınıfa** ayrılarak oluşturuldu. **'y'** dizisi, **0'dan 3'e** kadar rastgele tam sayılar içerir ve bu sayılar **3 sınıfı** temsil eder.
# - Burada kullanılan **"texture"**, genellikle bir **nesnenin** veya **yüzeyin dokusal özelliklerini** tanımlamak için kullanılıyor. Bu özellikler, yüzeyin pürüzlülüğü, düzensizliği, deseni veya diğer dokusal nitelikleri gibi görsel veya dokunsal özellikler olabilir. Örneğin, bir uzay aracının bir yüzeyindeki dokusal özellikler, yüzeydeki çizgilerin yoğunluğu veya renklerin değişkenliği gibi şeyler olabilir.

# In[6]:


import numpy as np
import pandas as pd

# Veri seti boyutlarını belirleme
n_samples = 400
n_features = 10

# Random özellikler ve sınıflar oluşturma
X = np.random.rand(n_samples, n_features)
y = np.random.randint(low=0, high=3, size=n_samples)

# Pandas DataFrame oluşturma
data = pd.DataFrame(X, columns=['feature_'+str(i) for i in range(n_features)])
data['class'] = y

# Özelliklerin yeni isimlerini tanımlama
new_feature_names = {
    'feature_0': 'x_coordinate',
    'feature_1': 'y_coordinate',
    'feature_2': 'z_coordinate',
    'feature_3': 'brightness',
    'feature_4': 'distance',
    'feature_5': 'spectral_intensity',
    'feature_6': 'texture_1',
    'feature_7': 'texture_2',
    'feature_8': 'texture_3',
    'feature_9': 'texture_4'
}

# Yeni özellik isimlerini Pandas DataFrame'e uygulama
data.rename(columns=new_feature_names, inplace=True)

# Veri setini CSV dosyası olarak kaydetme
data.to_csv('/Users/lenovo/Desktop/Soru_3.csv', index=False)


# #### A)	Naive Bayes sınıflandırıcı ile eğitim seti kullanarak oluşturduğunuz modeli test setine uygulayarak karmaşıklık matrisinin çıktısını yazınız.
# - İlk olarak gerekli **kütüphaneleri** ve **veri setini(dataset.csv)** yüklüyoruz.
# - Daha sonra veri setini **eğitim** ve **test** setlerine ayırıyoruz. Veri setinin **%80'i eğitim seti**, **%20'si** ise **test seti** olarak kullanılacak.
# - **Naive Bayes** sınıflandırıcısını oluşturuyoruz.
# - Son olarak, **test setini** kullanarak **sınıflandırma** yapacağız ve **karmaşıklık matrisini** oluşturacağız.

# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# dataset yükleme
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Hasan_Ali_Çalışkan_Soru_3.csv")

# özellikler ve sınıfların ayarlanması
X = df.drop('class', axis=1)
y = df['class']

# veri setini eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Naive Bayes modelini eğitme
nb = GaussianNB()
nb.fit(X_train, y_train)

# train ve test setleri için tahminleri alın
y_train_pred = nb.predict(X_train)
y_test_pred = nb.predict(X_test)

# Karmaşıklık matrisi hesaplama 
cm = confusion_matrix(y_test, y_test_pred)
print("Karmaşıklık Matrisi:\n", cm)


# #### B)	Sisteminizin başarısını aşağıda verilen metriklere göre analiz ederek ayrık ve sürekli değişkenlerin sistem performansına etkisini sınıf bazında sınıflandırma başarını göz önüne alarak açıklayınız.

# - İlk olarak, pandas kütüphanesi kullanılarak veri seti yüklenir ve özellikler (X) ve sınıflar (y) ayrılır.
# - Daha sonra, veri seti train_test_split () fonksiyonu kullanılarak eğitim ve test setlerine ayrılır.
# - GaussianNB () sınıflandırıcısı ile model eğitilir ve train ve test setleri için tahminler alınır.
# - accuracy_score (), recall_score (), f1_score () ve precision_score () fonksiyonları kullanılarak sınıflandırma performansı ölçülür ve ekrana yazdırılır.
# - Sonra, confusion_matrix () fonksiyonu kullanılarak bir karmaşıklık matrisi oluşturulur. Bu matris, sınıflandırmanın doğruluğunu daha ayrıntılı bir şekilde analiz etmek için kullanılır. 
# - Confusion matrix üzerinden sensitivity (duyarlılık) ve specificity (özgüllük) değerleri hesaplanır ve ekrana yazdırılır.

# In[8]:


from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

# Dataset yükleme
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Hasan_Ali_Çalışkan_Soru_3.csv")

# Özellikler ve sınıfların ayrılması
X = df.drop('class', axis=1)
y = df['class']

# Veri setinin eğitim ve test setlerine ayrılması
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Naive Bayes modeli
nb = GaussianNB()
nb.fit(X_train, y_train)

# Train ve test setleri için tahminlerin alınması
y_train_pred = nb.predict(X_train)
y_test_pred = nb.predict(X_test)

# Accuracy, recall, f-measure, precision ve classification report değerleri
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_test_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_test_pred, average='weighted')))
print("F-measure: {:.2f}".format(f1_score(y_test, y_test_pred, average='weighted')))
print("Precision: {:.2f}".format(precision_score(y_test, y_test_pred, average='weighted')))

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)

# Sensitivity
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

# Specificity
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])

print("Sensitivity: ", sensitivity)
print("Specificity: ", specificity)


# In[ ]:




