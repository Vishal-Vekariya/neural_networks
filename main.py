from src.data.make_dataset import load_and_preprocess_data
from src.features.build_features import create_dummy_vars
from src.visualization.visualize import scattor_plot, loss_curve, sub_plot
from src.models.train_model import neural_network
from src.models.predict_model import evaluate_model

if __name__ == "__main__":
    # Load and preprocess the data
    data_path = "data/raw/Admission.csv"
    data = load_and_preprocess_data(data_path)
    
    x, y = create_dummy_vars(data)
    
    MLP, Xtest, ytest,grid_p,Xtrain = neural_network(x,y)
    
    loss_curve(MLP)
    
    confusion_mat,accuracy,best_pa, grid_s,grid_sc = evaluate_model(MLP, Xtest, ytest, grid_p)
    print(f"Confusion Matrix:\n{confusion_mat}")
    print(f"Accuracy: {accuracy}")
    print(f"Best parameter: {best_pa}")
    print(f"Best Score: {grid_s}")
    print(f"Estimator: {grid_sc}")
    
    scattor_plot(data)
    sub_plot(data,Xtrain)