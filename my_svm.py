import pickle
import os
import cv2
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
import numpy as np
import tqdm
from SVM import SVM
from sklearn.decomposition import PCA


 
class Classifier(object):
    def __init__(self, filePath):
        self.filePath = filePath
 
    def unpickle(self, file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict
 
    def get_data(self):
        TrainData = None
        TestData = []
        for b in range(1,6):
            f = os.path.join(self.filePath, 'data_batch_%d' % (b, ))
            data = self.unpickle(f)
            train = np.reshape(data[b'data'], (10000, 3072))
            labels = np.reshape(data[b'labels'], (10000, 1))
            if TrainData is None:
                TrainData = np.concatenate([train, labels],axis=1)
            else:
                temp = np.concatenate([train, labels],axis=1)
                TrainData = np.concatenate([TrainData, temp],axis=0)
        f = os.path.join(self.filePath,'test_batch')
        data = self.unpickle(f)
        test = np.reshape(data[b'data'], (10000, 3072))
        labels = np.reshape(data[b'labels'], (10000, 1))
        TestData= np.concatenate([test, labels],axis=1)
 
        print("data read finished!")
        return TrainData, TestData
 
 
    def classification(self, train_feat, test_feat):

        print("Training my SVM Classifier.")
        
        train_data = train_feat[:, :-1]
        train_label =  train_feat[:, -1]
        test_data = test_feat[:,:-1]
        test_label = test_feat[:, -1]
 
        mean_train = np.mean(train_data, axis=0)
        train_data = train_data - mean_train
        mean_test = np.mean(test_data, axis=0)
        test_data = test_data - mean_test

        svm = SVM()
        loss_hist = svm.train(train_data, train_label, learning_rate=3e-7, reg=2e4,
                    num_iters=1800, verbose=False)
        y_train_pred = svm.predict(train_data)
        y_test_pred = svm.predict(test_data)
        y_train_acc = np.mean(y_train_pred==train_label)
        y_test_acc = np.mean(y_test_pred==test_label)
        print('My SVM train accuracy: %f test accuracy: %f' % (y_train_acc, y_test_acc))
   
    
    def PCA_SVM(self,train_feat, test_feat):
        
        print("Training my SVM Classifier with PCA.")
        train_data = train_feat[:, :-1]
        train_label =  train_feat[:, -1]
        test_data = test_feat[:,:-1]
        test_label = test_feat[:, -1]
 
        mean_train = np.mean(train_data, axis=0)
        train_data = train_data - mean_train
        mean_test = np.mean(test_data, axis=0)
        test_data = test_data - mean_test

        x_shape=[]		
        test_scores = []		
        train_scores = []
        ratio = np.linspace(0.6,0.999,10)
        for i in ratio:
            pca = PCA(i)
            svm = SVM()
            x_train = pca.fit_transform(train_data)
            x_test = pca.transform(test_data)
            

            loss_hist = svm.train(x_train, train_label, learning_rate=3e-7, reg=2e4,
                    num_iters=1800, verbose=False)

            y_train_pred = svm.predict(x_train)
            y_test_pred = svm.predict(x_test)
            train_acc = np.mean(y_train_pred==train_label)
            test_acc = np.mean(y_test_pred==test_label)
            
            print("ratio{}-----use {} features: get {} acc on test and {} acc on train".format( i,x_train.shape[1], test_acc, train_acc))
            
            x_shape.append(x_train.shape[1])
            test_scores.append(test_acc)
            train_scores.append(train_acc)

        plt.plot(x_shape,test_scores)
        plt.xlabel('number of features')
        plt.ylabel('test accuracy')
        plt.show()


    def run(self):
        TrainData, TestData = self.get_data()

        self.classification(TrainData, TestData)
        self.PCA_SVM(TrainData, TestData)
 



if __name__ == '__main__':
    filePath = r'cifar-10-python\cifar-10-batches-py'
    cf = Classifier(filePath)
    cf.run()