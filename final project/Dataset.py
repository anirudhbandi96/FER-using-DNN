import numpy as np
import os
class data(object):
    def __init__(self ):
        self.features = None
        self.labels = None
        
class image_data(object):
    def __init__(self):
        print('image_data object created')
        self.training = data()
        self.testing = data()
        self.classes  = 0
    def data_details(self):
        print(self.training.features.size)
        print(self.training.labels.size)

        print(self.testing.features.size)
        print(self.testing.labels.size)
        
    def read_textFile(self , location,filename):
        os.chdir(location)
        file = open(filename, 'r')
        features = []
        labels = []
        a = []
        flag = 1
        k = 0
        for line in file:
            # k += 1
            line = line.strip('\n')
            line = line.strip()
            row = line.split(' ')
            l = len(row)
            if '' not in row: 

                if l > 2:
                    #print(k)
                    for i in range(l):
                        row[i] = float(row[i])
                    #print(len(row))
                    a.append(row)
                else:
                    labels.append(int(row[0]))
                    flag = 0
            else:
                if flag:
                    features.append(a)
                    a = []
            #this is to make sure that the last images's data is obtained
        if flag:
            features.append(a)
            a=[]
        if len(features)!= 0:
            return np.array(features , dtype = float)
        else:
            return labels
    def read_jaffe_data(self):
        self.classes = 7
        print('number of classes: ' , self.classes)
        self.read_feature_data('/Users/Anirudh/Desktop/project','training_features.txt','testing_features.txt')
        self.read_label_data('/Users/Anirudh/Desktop/project','training_labels.txt','testing_labels.txt')
        

    def read_ck_data(self):
        self.classes = 6
        self.read_feature_data('/Users/Anirudh/Desktop/project','ck_train_features.txt','ck_test_features.txt')
        self.read_label_data('/Users/Anirudh/Desktop/project','ck_train_labels.txt','ck_test_labels.txt')
        
        
        

    def read_feature_data(self,base,train_filename , test_filename):
        print('reading training features....')
        self.training.features = self.read_textFile(base,train_filename)
        print('reading testing features....')
        self.testing.features = self.read_textFile(base, test_filename)
        
    def read_label_data(self , base , train_filename,test_filename):
        print('reading training labels....')
        self.training.labels = self.one_hot_data(self.read_textFile(base, train_filename))
        print('reading testing labels....')
        self.testing.labels = self.one_hot_data(self.read_textFile(base, test_filename))

    def one_hot_data(self , vector):
        #number of classes (should make it a variable later)
        
        b = np.zeros(shape=[len(vector),self.classes])
        for i in range(len(vector)):
            b[i , vector[i]-1] = 1
        return b

    def print_data(self):
        #print(self.training.features[-1])
        print(self.training.labels)
        print(self.testing.labels)
    def get_training_samples_length(self):
        #print(len(self.training.labels))
        return len(self.training.labels)
    
            
        
        
    
