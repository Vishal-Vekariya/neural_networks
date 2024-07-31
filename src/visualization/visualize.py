import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np


def scattor_plot (data):
    
    plt.figure(figsize=(15,8))
    sns.scatterplot(data=data, 
           x='GRE_Score', 
           y='TOEFL_Score', 
           hue='Admit_Chance')
    plt.title('Scattor Plot', fontsize=16)
    plt.show()
    

def loss_curve (MLP):
       loss_values = MLP.loss_curve_
       plt.figure(figsize=(10, 6))
       plt.plot(loss_values, label='Loss', color='blue')
       plt.title('Loss Curve')
       plt.xlabel('Iterations')
       plt.ylabel('Loss')
       plt.legend()
       plt.grid(True)
       plt.show()
       
def sub_plot(data,Xtrain):
       plt.subplot(2,2,1)
       sns.distplot(data['GRE_Score'])

       plt.subplot(2,2,2)
       sns.distplot(Xtrain[:,0])

       plt.subplot(2,2,3)
       sns.distplot(data['TOEFL_Score'])

       plt.subplot(2,2,4)
       sns.distplot(Xtrain[:,1])

       plt.show()