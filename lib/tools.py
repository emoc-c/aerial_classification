def repartition(y):
    '''
    repartition : plot the repartition of the images
    inputs:
    y=the images labels
    outputs:
    
    '''
    labels={}
    for yi in y:
        if yi in labels:
            labels[yi]+=1
        else :
            labels[yi]=1
    Y=np.array(list(labels.values()))
    X=np.arange(0,len(Y),1)
    
    Y_re=Y.reshape(len(Y),1)
    X_re=X.reshape(len(Y),1)
    plt.figure(figsize=(8, 6), dpi=80)
    for i in range(len(Y)):  
        plt.bar(X_re[i],Y_re[i],label=list(labels.keys())[i])
    plt.legend()

    
import matplotlib.pyplot as plt
import numpy as np
    

def true_pred(y_pred,y_true):
    '''
    true_pred : plot the pourcentage of true pedictions for each class
    inputs :
    y_pred : the predictions
    y_true : the ground truth
    outputs :
    
    '''
    num={}
    score={}
    for y in range(len(y_pred)):
        if y_pred[y] in num:
            num[y_pred[y]]+=1
        else :
            num[y_pred[y]]=1
            score[y_pred[y]]=0
        if y_pred[y]==y_true[y] :
            score[y_pred[y]]+=1
    Y=np.array(list(score.values()))/np.array(list(num.values()))
    X=np.arange(0,len(Y),1)
    
    Y_re=Y.reshape(len(Y),1)
    X_re=X.reshape(len(Y),1)
    plt.figure(figsize=(8, 6), dpi=80)
    for i in range(len(Y)):  
        plt.bar(X_re[i],Y_re[i],label=list(score.keys())[i])
    plt.legend()
     
def score(y_pred,y_true):
    '''score : calculate the accuracy of a model
    inputs :
    y_pred : predicted labels
    y_true : ground truth
    outputs :
    the accuracy'''
    score=0.
    for i in range(len(y_pred)):
        score+= y_pred[i]==y_true[i]
    return score/len(y_pred)



def kfold(data,labels,nb_fold,model):
    '''kfold : prefom a kfold cross validation
    inputs :
    data : the data we want to work on
    labels : the true labels of the data
    nb_fold : the number of folds
    model : the model to train and test | x_train,ytrain,x_test-->pred
    outputs : 
    the average accuracy'''
        
    size=int(len(data)/nb_fold)
    acc=0
    for i in range(nb_fold):
        print('fold '+str(i+1)+'/'+str(nb_fold))
        test_set=data[i*size:(i+1)*size]
        train_set=data[(i+1)*size:]
        label_set=labels[(i+1)*size:]
        if i!=0:
            train_set=list(data[:i*size])+list(data[(i+1)*size:])
            label_set=list(labels[:i*size])+list(labels[(i+1)*size:])
        y_pred=model(train_set,label_set,test_set)
        acc+=score(y_pred,labels[i*size:(i+1)*size])
    return acc/nb_fold

