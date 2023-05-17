#!/usr/bin/env python
# coding: utf-8

# In[34]:


# Soru 4)
# Kendinizin belirleyeceğiniz bir veri seti kullanarak random forest algoritması ile bütün özelliklerinizi kullanarak sınıflandırma başarısını hesaplayınız.

# CEVAP: 

# İlk olarak gerekli kütüphaneleri ekliyoruz.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# dataset yükleme
# Bu dataseti; 10 özellikli, 400 veri örneklidir.
df = pd.read_csv(r"C:\Users\lenovo\Desktop\Soru_4.csv")

# Daha sonra, veri kümesi X_train, X_test, y_train ve y_test adlı dört parçaya bölünür, böylece eğitim verileri ve test verileri ayrılabilir.
X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'], test_size=0.25)

# RandomForestClassifier sınıfı kullanılarak bir RandomForest sınıflandırıcı modeli oluşturuyoruz. 
# n_estimators parametresi, RandomForest'da kullanılacak ağaç sayısını belirlerken max_depth parametresi, her ağaçta bulunan maksimum derinlik sayısını belirler.
#  Model eğitildikten sonra, predict() fonksiyonu kullanılarak test verilerinde tahminler yapıyoruz.
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                                     min_samples_split=2, min_samples_leaf=1, 
                                     min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                     max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                     bootstrap=True, oob_score=False, n_jobs=None, 
                                     random_state=None, verbose=0,warm_start=False, 
                                     class_weight=None, ccp_alpha=0.0, max_samples=None)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# Burada özelliklerin önem derecelerini görüyoruz. (Veri setinde toplam 10 özellik var ve burada 10'unu da görüyoruz.)
plt.bar(x=feature_scores.index, height=feature_scores.values)
plt.xticks(rotation=90)
plt.show()

# Accuracy, recall, f-measure, precision ve classification report değerleri
print("Accuracy: {:.2f}".format(accuracy_score(y_test, y_pred)))
print("Recall: {:.2f}".format(recall_score(y_test, y_pred, average='weighted')))
print("F-measure: {:.2f}".format(f1_score(y_test, y_pred, average='weighted')))
print("Precision: {:.2f}".format(precision_score(y_test, y_pred, average='weighted')))


# Öznitelik önem sıralaması
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

X_train_reduced = X_train[feature_scores.index[2:]]
X_test_reduced = X_test[feature_scores.index[2:]]

clf_reduced.fit(X_train_reduced, y_train)

# Son olarak, daha önceki adımda oluşturulan yeni veri kümesi kullanılarak yeniden sınıflandırma yapıyoruz ve sınıflandırmanın performansı, 
# classification_report() fonksiyonu kullanılarak hesaplıyoruz ve ekrana yazdırıyoruz.
y_pred = clf_reduced.predict(X_test_reduced)
print(classification_report(y_test, y_pred))

print()

# Şimdi önem derecesine göre sıraladıktan sonra k>2 olacak şekilde en az etki eden 4 özelliği çıkarıp tekrar test ediyoruz.
# Öznitelik önem sıralaması
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# En az etki eden 4 özelliği çıkarıyoruz
X_train_reduced = X_train[feature_scores.index[4:]]
X_test_reduced = X_test[feature_scores.index[4:]]

# Yeni özelliklerle bir model oluşturuyoruz
clf_reduced = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None,
                                      min_samples_split=2, min_samples_leaf=1, 
                                      min_weight_fraction_leaf=0.0, max_features='sqrt', 
                                      max_leaf_nodes=None, min_impurity_decrease=0.0, 
                                      bootstrap=True, oob_score=False, n_jobs=None, 
                                      random_state=None, verbose=0,warm_start=False, 
                                      class_weight=None, ccp_alpha=0.0, max_samples=None)

clf_reduced.fit(X_train_reduced, y_train)

# Yeniden sınıflandırma yapıp performansı ölçüyoruz.
y_pred = clf_reduced.predict(X_test_reduced)
print(classification_report(y_test, y_pred))


# In[ ]:




