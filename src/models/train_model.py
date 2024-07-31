from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def neural_network(x,y):
    xtrain, xtest, ytrain, ytest =  train_test_split(x, y, test_size=0.2, random_state=123)
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    Xtest = scaler.transform(xtest)
    MLP = MLPClassifier(hidden_layer_sizes=(3), batch_size=50, max_iter=100, random_state=123)
    MLP.fit(Xtrain,ytrain)
    params = {'batch_size':[20, 30, 40, 50],
          'hidden_layer_sizes':[(2,),(3,),(3,2)],
         'max_iter':[50, 70, 100]}
    grid = GridSearchCV(MLP, params, cv=10, scoring='accuracy')
    grid_p = grid.fit(x, y)
    
    
    return MLP, Xtest, ytest,grid_p,Xtrain