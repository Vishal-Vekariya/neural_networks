from sklearn.metrics import confusion_matrix, accuracy_score

def evaluate_model(MLP, Xtest, ytest, grid_p):
    ypred = MLP.predict(Xtest)
    confusion_mat = confusion_matrix(ytest, ypred)
    accuracy = accuracy_score(ytest, ypred)
    best_pa = grid_p.best_params_
    grid_s = grid_p.best_score_
    grid_sc=  grid_p.estimator
    return confusion_mat,accuracy, best_pa, grid_s,grid_sc