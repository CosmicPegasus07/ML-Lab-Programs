import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

iris=load_iris()

X=pd.DataFrame(iris.data)
X.columns=['Sepal_Length','Sepal_Width','Petal_Length','Petal_Width']
Y=pd.DataFrame(iris.target)
Y.columns=['Targets']
print(X)
print(Y)

colormap=np.array(['red','lime','black'])

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[Y.Targets],s=40)
plt.title('Real Clustering')

model1=KMeans(n_clusters=3)
model1.fit(X)

plt.subplot(1,2,2)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model1.labels_],s=40)
plt.title('K-Means Clustering')
plt.show()

model2=GaussianMixture(n_components=3)
model2.fit(X)

plt.subplot(1,2,1)
plt.scatter(X.Petal_Length,X.Petal_Width,c=colormap[model2.predict(X)],s=40)
plt.title('K-Means Clustering')
plt.show()

print('Accuracy of KMeans is',metrics.accuracy_score(Y,model1.labels_))
print('Accuracy of EM is',metrics.accuracy_score(Y,model2.predict(X)))