import numpy as np
import lib.tile_model as t
from sklearn.decomposition import PCA


def get_freq(labels,data):
    '''get_freq : return the percentage of each element of labels
    inputs :
    labels : possible elements
    data : list of labels elements were we want to perform
    outputs :
    the list of percentages
    '''
    d=dict(zip(labels,[0]*len(labels)))
    for label in data:
        d[label]+=1
    return np.array(list(d.values()))/float(len(data))


def tile_train_c(trained_model,to_train_model,x_train,y_train,labels=["beach","chaparral","cloud","desert","forest","island","lake","meadow","mountain","river","sea","snowberg","wetland"],tile_size=64,offset=64,pc=False,pca=None):
    '''tile_train_c : train the second model, we devide the image in tiles, for each image we predict the label of each tile, after we get the percentage of each zone for each image and we train the second model on those percentages
    inputs:
    trained_model : trained tile recognition model
    to_train_model : percentage classification model to train
    x_train : train set for the second model
    y_train : labels of the train set
    labels : possible labels
    tile_size : size of the sub_images
    offset : indent between the subimages
    pc : true = apply pca
    pca : the trained pca to apply
    outputs : 
    the trained model
    '''
    features=[]
    for imgi in x_train:
        tile=t.tile_img(imgi,tile_size,offset)
        if pc :
            tiler=pca.transform(tile)
            y=trained_model.predict(tiler)
        else :
            y=trained_model.predict(tile)
        features.append(get_freq(labels,y))
    to_train_model.fit(features,y_train)
    return to_train_model


def tile_test_c(trained_m1,trained_m2,x_test,labels=["beach","chaparral","cloud","desert","forest","island","lake","meadow","mountain","river","sea","snowberg","wetland"],tile_size=50,offset=30,pc=False,pca=None):
    '''tile_test_c : test the cascade learning model
    inputs :
    trained_m1 : trained tile recognition model
    trained_m2 : trained percentage classification model
    labels : possible labels
    tile_size : size of the sub_images
    offset : indent between the subimages
    pc : true = apply pca
    pca : the trained pca to apply
    outputs : 
    the trained model
    '''
    y_pred=[]
    for imgi in x_test:
        tile=t.tile_img(imgi,tile_size,offset)
        if pc :
            tiler=pca.transform(tile)
            y=trained_m1.predict(tiler)
        else :
            y=trained_m1.predict(tile)
        #print(y)
        #plt.imshow(np.array(tile[0]).reshape(64,64,3).astype(float)/255)
        y_pred.append(get_freq(labels,y))
    return trained_m2.predict(y_pred)


from sklearn.model_selection import train_test_split

def train_test_c(x_train,y_train,x_test,m1,m2,tile_size=8,offset=8,pc=True,nb_cpnt=50):
    '''train_test_c : the model for the kfold function
    inputs :
    x_train : the train set
    y_train : the labels of thr train set
    x_test : the test set
    m1 : the first model (on tiles)
    m2 : the second model ( on percentages)
    tile_size : the size of thr tiles
    offset : indent between the subimages
    pc : true if pca
    nb_cpnt : the nb_components of the pca
    outputs :
    the predictions on the test set'''
    x_train1,x_train2,y_train1,y_train2=train_test_split(x_train, y_train, test_size=0.5)
    r=t.tile_train(m1,x_train1,y_train1,tile_size=tile_size,offset=offset,pca=pc,n_cpnt=nb_cpnt)
    trained_m1=None
    pca=None
    if pc :
        trained_m1,pca=r
    else :
        trained_m1=r
    trained_m2=tile_train_c(trained_m1,m2,x_train2,y_train2,tile_size=tile_size,offset=offset,pc=pc,pca=pca)
    return tile_test_c(trained_m1,trained_m2,x_test,tile_size=tile_size,offset=offset,pc=pc,pca=pca)
