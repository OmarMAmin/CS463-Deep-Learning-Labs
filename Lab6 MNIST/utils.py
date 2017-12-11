def history_appending(history1,history2):
    history1.history['acc'] += history2.history['acc']
    history1.history['val_acc'] += history2.history['val_acc']
    history1.history['val_loss'] += history2.history['val_loss']
    history1.history['loss'] += history2.history['loss']
    return history1

def history_from_list(history_list):
    for i in range(len(history_list)-1):
        history_list[0].history['acc'] += history_list[i+1].history['acc']
        history_list[0].history['val_acc'] += history_list[i+1].history['val_acc']
        history_list[0].history['val_loss'] += history_list[i+1].history['val_loss']
        history_list[0].history['loss'] += history_list[i+1].history['loss']
    return history_list[0]
    
def to_kaggle_csv(matrix, header,filename):
    frame = pd.DataFrame(data = matrix,columns=header)
    frame.to_csv(path_or_buf  = filename,index = False,sep =',')
    return frame

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
def load_train_valid_test_data(train_file,test_file,validation_percentage):
    train_data = np.genfromtxt(delimiter=',',fname=train_file,skip_header=True)
    test_data = np.genfromtxt(delimiter=',',fname=test_file,skip_header=True)
    train_y = train_data[:,0]
    train_x = train_data[:,1:]
    train_x = train_x.reshape((train_data.shape[0],28,28,1))
    enc = preprocessing.OneHotEncoder()
    enc.fit(train_y.reshape((train_y.shape[0],1)))
    train_y = enc.transform(train_y.reshape((train_y.shape[0],1))).toarray()
    train_x /= 255
    test_x = test_data.reshape((test_data.shape[0],28,28,1))
    test_x  = test_x/255
    train_x = train_x.astype('float32')
    test_x = test_x.astype('float32')
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)
    
    return train_x, valid_x, test_x, train_y, valid_y

def generate_submission(model,data,filename,file_columns = None):
    predictions = model.predict(data)
    result = predictions.argmax(axis =1 )
    imgs_count = data.shape[0]
    result = result.reshape((imgs_count,1))
    ids = (np.arange(imgs_count)+1).reshape((imgs_count,1))
    result = np.hstack((ids,result))
    m = to_kaggle_csv(result,file_columns, filename)

def ensemble_submission(models,test_data,filename,file_columns = None):
    predictions = model[0].predict(data)
    for idx in range(len(models - 1)):
        predictions = predictions + model[idx+1].predict(data)
    result = predictions.argmax(axis =1)
    imgs_count = data.shape[0]
    result = result.reshape((imgs_count,1))
    ids = (np.arange(imgs_count)+1).reshape((imgs_count,1))
    result = np.hstack((ids,result))
    m = to_kaggle_csv(result,file_columns, filename)
    
def validate_preprocessing(train_x, valid_x, test_x, train_y, valid_y):
    print('max train',train_x.max())
    print('max valid',train_x.max())
    print('max test',train_x.max())
    
    print('train_x shape',train_x.shape)
    print('valid_x shape',valid_x.shape)
    print('test_x shape',test_x.shape)
    print('train_y shape',train_y.shape)
    print('valid_y shape',valid_y.shape)
