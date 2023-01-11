from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
#Load dataset
iris=datasets.load_iris()
print('Iris dataset loaded')
#Split dataset into train and test samples
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.1)
#Print label no. and their names
for i in range(len(iris.target_names)):
    print('Label',i,'-',str(iris.target_names[i]))
#Create object of KNN classifier
classifier=KNeighborsClassifier(n_neighbors=1)
#Perform training
classifier.fit(x_train,y_train)
#Perform testing
y_pred=classifier.predict(x_test)
#Display the results
print('Results of Classification using K-nn with K=1')
for r in range(0,len(x_test)):
    print('Sample:',str(x_test[r]),'Actual-label:',str(y_test[r]),'Predicted label:',str(y_pred[r]))
print('Classification Accuracy:',classifier.score(x_test,y_test))