from numpy import array,dot,exp,max,sum,subtract,log,outer,save,load
from numpy.random import uniform
from numpy.linalg import norm
# from os import listdir
# from re import sub

settings=[2,10,50,0.01] # windowSize, dimension, epochs, learningRate

class wordToVector():
    def __init__(self):
        # necessary configuration
        self.window=3 # word and context words
        self.dimension=800#10 number of similarities, categories
        self.epochs=50 # iterations of self correction
        self.learningRate=0.01 # drift of self corrections

        # import wordCount as stringList and integerIndex
        paper=open(r'F:\\文档\\data\\output\\english\\1grandCount.txt','r',encoding='utf-8')
        self.vocabularyList=paper.readlines()
        self.indexList=[]
        for index,lineString in enumerate(self.vocabularyList):
            coupling=lineString.split()
            self.vocabularyList[index]=coupling[0] # lineString.split()[0]
            self.indexList.append(index)

        self.vectorCount=len(self.vocabularyList) # individual word count

        # raw corpus as text
        paper=open(r'F:\\文档\\data\\output\\english\\2grandCorpus.txt','r',encoding='utf-8')
        reading=paper.readline()
        self.paragraphList=[]
        while reading:
            #for element in self.skipList:reading=sub(element,' ',reading)
            self.paragraphList.append(reading)
            reading=paper.readline()

    def generateTrainingData(self):
        trainingData=[]
        for paragraphString in self.paragraphList:
            sentenceList=paragraphString.split('.')
            for sentenceString in sentenceList:
                if not sentenceString.strip():continue
                wordList=sentenceString.split()
                sentenceLength=len(wordList)
                for index,wordString in enumerate(wordList):
                    wordNode=self.wordToOneHot(wordString)
                    wordContext=[]
                    for dexin in range(index-self.window,index+self.window+1):
                        if dexin!=index and dexin <=sentenceLength-1 and dexin>=0:
                            wordContext.append(self.wordToOneHot(wordList[dexin]))
                wordNode=[index for index,entry in enumerate(wordNode) if entry==1.0]
                wordContext=[[index for index,subentry in enumerate(entry) if subentry==1.0][0] for entry in wordContext]
                trainingData.append([wordNode,wordContext]) # data is oneHot of target and list for oneHot of context
        return array(trainingData)
        
    def wordToOneHot(self,word): # form of encoding [0,0,0,1,0,0]
        wordVector=[0 for i in range(0,self.vectorCount)]
        # wordIndex=self.wordIndex[word]
        # wordIndex=self.vocabularyList.index(word)
        wordVector[self.vocabularyList.index(word)]=1
        return wordVector

    def train(self,trainingData):
        # self.weightOne=array(getW1)
        # self.vocabularyList,self.vectorCount,self.indexList=inputOne,inputTwo,inputThree
        self.weightOne=uniform(-1,1,(self.vectorCount,self.dimension))
        self.weightTwo=uniform(-1,1,(self.dimension,self.vectorCount))

        for iteration in range(self.epochs):
            self.loss=0
            for wordNode,wordContext in trainingData: # wordNode w_t
                yPrediction,hiddenLayer,outputLayer=self.forwardPass(wordNode)

                errorInput=sum([subtract(yPrediction,word) for word in wordContext],axis=0)
                self.backprop(errorInput,hiddenLayer,wordNode)

                # onehot of context indexed for outerlayer
                # list of hot indexes
                self.loss+=-sum([outputLayer[word.index(1)] for word in wordContext])+len(wordContext)*log(sum(exp(outputLayer)))
            #print('Epoch:',i,"Loss:",self.loss)
        save(r'F:\\文档\\data\\output\\english\\training\\soundWeight.npy',self.weightOne)

    def backprop(self,error,hiddenLayer,x):
        if isinstance(error,float) and error==0.0:return
        weightTwoDelta=outer(hiddenLayer,error)
        # weightOneDelta=outer(x,dot(self.weightTwo,error.T))
        self.weightOne=self.weightOne-self.learningRate*(outer(x,dot(self.weightTwo,error.T)))
        # self.weightOne=self.weightOne-(self.learningRate*weightOneDelta)
        self.weightTwo=self.weightTwo-(self.learningRate*weightTwoDelta)

    def forwardPass(self,initialLayer):
        hiddenLayer=dot(self.weightOne.T,initialLayer) # initial layer onehot list
        outputLayer=dot(self.weightTwo.T,hiddenLayer) # hidden layer numpy (dimension,)
        y_c=self.softmax(outputLayer) # output layer numpy (vectorCount,)
        return y_c,hiddenLayer,outputLayer
    
    def softmax(self,initialLayer):
        eOfx=exp(initialLayer-max(initialLayer))
        return eOfx / eOfx.sum(axis=0)
    
    def vectorizeWord(self,wordString):
        wordIndex=self.indexList[self.vocabularyList.index(wordString)]
        vectoredWord=self.weightOne[wordIndex]
        return vectoredWord

    def similarVector(self,wordString,topInteger):
        vectorWeightOne=self.vectorizeWord(wordString)
        similarWord={}

        for index in range(self.vectorCount):
            vectorWeightTwo=self.weightOne[index]
            thetaSummation=dot(vectorWeightOne,vectorWeightTwo)
            thetaDen=norm(vectorWeightOne)*norm(vectorWeightTwo)
            theta=thetaSummation/thetaDen

            wordString=self.vocabularyList[index]
            similarWord[wordString]=theta
        sortedWords=sorted(similarWord.items(),key=lambda kv:kv[1], reverse=True)

        for word,similar in sortedWords[:topInteger]:
            print(word,similar)

vectorize=wordToVector()
trainingData=vectorize.generateTrainingData()
# save(r'F:\\文档\\data\\output\\english\\training\\hotModel.npy',trainingData)
# trainingData=load(r'F:\\文档\\data\\output\\english\\training\\hotModel.npy',allow_pickle=True)
vectorize.train(trainingData)
# vectorize.similarVector("heaven",10)
pass

# heaven 1.0
# favour 0.9189760690670926
# inspection 0.9157220039490127
# bestowed 0.9150858265818741
# plea 0.910679870564455
# delicate 0.9015822578984337
# loya 0.8895275787968661
# belonged 0.88531871661095
# physiognomist 0.8782884864772055
# societal 0.8736370531350078


# def encoding2(self):
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import pandas as pd

#     # Load the MNIST dataset
#     from sklearn.datasets import fetch_openml
#     mnist = fetch_openml('mnist_784')
#     X = mnist.data / 255.
#     X = pd.DataFrame(X)  # reshape each instance to a 28x28 array

#     # Define the architecture of the autoencoder
#     input_size = 784   # 28*28
#     hidden_size = 64   # number of neurons in the hidden layer
#     output_size = 784  # reconstruction of the input

#     # Initialize the weights of the autoencoder
#     W1 = np.random.randn(input_size, hidden_size)
#     W2 = np.random.randn(hidden_size, output_size)

#     # Define the activation function (sigmoid)
#     def sigmoid(x):return 1 / (1 + np.exp(-x))

#     # Define the derivative of the activation function
#     def sigmoid_derivative(x):return sigmoid(x) * (1 - sigmoid(x))

#     # Define the learning rate and the number of epochs
#     learning_rate = 0.1
#     num_epochs = 1 # 1000

#     # Train the autoencoder
#     for epoch in range(num_epochs):
#         # Forward pass
#         X_flat = X.to_numpy().reshape((-1, input_size))  # convert to NumPy array and flatten each instance to a 1D array
#         hidden = sigmoid(np.dot(X_flat, W1))
#         output = sigmoid(np.dot(hidden, W2))
        
#         # Compute the loss and the gradients
#         loss = np.mean((output - X_flat)**2)
#         doutput = (output - X_flat) * sigmoid_derivative(output)
#         dhidden = np.dot(doutput, W2.T) * sigmoid_derivative(hidden)
        
#         # Update the weights
#         W2 -= learning_rate * np.dot(hidden.T, doutput)
#         W1 -= learning_rate * np.dot(X_flat.T, dhidden)
        
#         # Print the loss every 100 epochs
#         if epoch % 10 == 0: # 100 == 0
#             print(f"Epoch {epoch}: loss = {loss:.4f}")

#     # Test the autoencoder
#     X_test=X.to_numpy()[:10]# take the first 10 instances for testing
#     X_test_flat = X_test.reshape((-1, input_size))
#     hidden_test = sigmoid(np.dot(X_test_flat, W1))
#     reconstructed_test = sigmoid(np.dot(hidden_test, W2))
#     reconstructed_test = reconstructed_test.reshape((-1, 28, 28))  # reshape each instance back to a 28x28 array

#     # Plot some examples
#     n_examples = 10
#     plt.figure(figsize=(20, 4))
#     for i in range(n_examples):
#         # Original
#         ax = plt.subplot(2, n_examples, i + 1)
#         X_test=np.expand_dims(X_test,axis=0)
#         X_test=np.expand_dims(X_test,axis=0)
#         plt.imshow(X_test[i][:, :, 0], cmap='gray')
#         plt.axis('off')
#         # plt.gray()
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
        
#         # Reconstruction
#         ax = plt.subplot(2, n_examples, i + 1 + n_examples)
#         reconstructed_test=np.expand_dims(reconstructed_test,axis=0)
#         plt.imshow(reconstructed_test[i][:, :, 0], cmap='gray')
#         plt.axis('off')
#         # plt.gray()
#         # ax.get_xaxis().set_visible(False)
#         # ax.get_yaxis().set_visible(False)
#     plt.show()