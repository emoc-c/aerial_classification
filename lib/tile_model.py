from sklearn.decomposition import PCA
from statistics import mode

def tile(img,x,y,tile_size):
    '''tile: create a square subimage from img that start in (x,y) of size tile_size
    inputs:
    img : the initial image
    x : the x start position
    y : the y start position
    tile_size : the size of the subimage
    outputs:
    tile : the sub_image'''
    x=x*3
    tile=[]
    line_size=128*3
    for line in range(tile_size):
        for p in img[(line+y)*(line_size)+x:(line+y)*line_size+x+tile_size*3]:
            tile.append(p)
    return tile


def tile_img(img,tile_size,offset=32):
    '''tile_img: create all the square subimages of img of size tile_size with an indent of offset
    inputs:
    img : the initial image
    tile_size : the size of the subimages
    offset : the indent between the images
    outputs:
    tiles= the list of subimages'''
    tiles=[]
    line_size=128
    for y in range(0,line_size-tile_size+1,offset):
        for x in range(0,int((line_size*3-tile_size)/3+1),offset):
            tiles.append(tile(img,x,y,tile_size))
    if(offset!=tile_size):
        tiles.pop()
            
    return tiles


def tile_train(model,data,y,tile_size=64,offset=64,pca=True,n_cpnt=50):
    '''tile_train : train the tile recognition model
    inputs:
    model : the model to train
    data : the train set
    y : the labels
    tile_size : the size of the subimages
    offset : the indent netween the subimages
    pca : true = apply pca,false = no pca
    n_cpnt : the number of components you want to keep after pca
    outputs:
    model : the trained model
    pc = the trained pca'''
    tiles=[]
    labels=[]
    nb_tiles=0
    for i,imgi in enumerate(data):
        new=tile_img(imgi,tile_size,offset)
        tiles=tiles+new
        if(i==0):
            nb_tiles=len(new)
        labels=labels+[y[i]]*nb_tiles
    print("fiting time")
    if pca :
        pc=PCA(n_components=n_cpnt)
        tilesr=pc.fit_transform(tiles,labels)
        model.fit(tilesr,labels)
        return model,pc
    else :    
        model.fit(tiles,labels)
        return model
    
    
def most_common(List):
    '''most_common: return the most common element of a list
    inputs:
    List : a list
    outputs : 
    the most common element'''
    return(mode(List))

def tile_test(trained_model,x_test,tile_size=64,offset=64,pc=False,pca=None):
    '''tile_test: test the trained model,for each test image we devide it in tiles and predict its label with our trained model then we assign to the image the most common label among the tiles
    inputs:
    trained_model : the trained model
    x_test : the test set
    tile_size : the size of the subimages
    offset : the indent between subimages
    pc : true = apply pca,false = no pca
    pca = the trained pca
    outputs :
    y_pred : the predicted labels
    '''
    y_pred=[]
    for imgi in x_test:
        tile=tile_img(imgi,tile_size,offset)
        if pc :
            tiler=pca.transform(tile)
            y=trained_model.predict(tiler)
        else :
            y=trained_model.predict(tile)
            
        #print(y)
        #plt.imshow(np.array(tile[0]).reshape(64,64,3).astype(float)/255)
        y_pred.append(most_common(y))
        
    return y_pred