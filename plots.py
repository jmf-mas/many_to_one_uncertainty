import numpy as np
from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
from scipy.stats import pearsonr as pcc
from scipy.stats import spearmanr as scc
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

directory_data = "data/"
directory_outputs = "outputs/"
kdd = "kdd"
nsl = "nsl"
ids = "ids"
kitsune = "kitsune"
ciciot = "ciciot"
metrics = {"ent":-1, "std":-2}

def get_plots(i, metric, filename):
    
    y_pred_train_ids = np.loadtxt(directory_outputs + ids + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_ids = np.loadtxt(directory_outputs + ids + "_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_ids = np.loadtxt(directory_outputs + ids + "_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_nsl = np.loadtxt(directory_outputs + nsl + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_nsl = np.loadtxt(directory_outputs + nsl + "_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_nsl = np.loadtxt(directory_outputs + nsl +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_kdd = np.loadtxt(directory_outputs + kdd + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_kdd = np.loadtxt(directory_outputs + kdd +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_kdd = np.loadtxt(directory_outputs + kdd +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_kitsune = np.loadtxt(directory_outputs + kitsune + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_kitsune = np.loadtxt(directory_outputs + kitsune +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_kitsune = np.loadtxt(directory_outputs + kitsune +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_ciciot = np.loadtxt(directory_outputs + ciciot + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_ciciot = np.loadtxt(directory_outputs + ciciot +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_ciciot = np.loadtxt(directory_outputs + ciciot +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    
    XY_kdd_train = np.loadtxt(directory_data + kdd + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_kdd = XY_kdd_train[:, metrics[metric]]
    X_train_kdd = XY_kdd_train[:, :-2]
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_kdd = XY_kdd_val[:, metrics[metric]]
    X_val_kdd = XY_kdd_val[:, :-2]
    XY_kdd_test = np.loadtxt(directory_data + kdd +"_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_kdd = XY_kdd_test[:, metrics[metric]]
    X_test_kdd = XY_kdd_test[:, :-2]
        
    XY_nsl_train = np.loadtxt(directory_data + nsl + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_nsl = XY_nsl_train[:, metrics[metric]]
    X_train_nsl = XY_nsl_train[:, :-2]
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_nsl = XY_nsl_val[:, metrics[metric]]
    X_val_nsl = XY_nsl_val[:, :-2]
    XY_nsl_test = np.loadtxt(directory_data + nsl + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_nsl = XY_nsl_test[:, metrics[metric]]
    X_test_nsl = XY_nsl_test[:, :-2]
        
    XY_ids_train = np.loadtxt(directory_data + ids + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_ids = XY_ids_train[:, metrics[metric]]
    X_train_ids = XY_ids_train[:, :-2]
    XY_ids_val = np.loadtxt(directory_data + ids + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_ids = XY_ids_val[:, metrics[metric]]
    X_val_ids = XY_ids_val[:, :-2]
    XY_ids_test = np.loadtxt(directory_data + ids + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_ids = XY_ids_test[:, metrics[metric]]
    X_test_ids = XY_ids_test[:, :-2]
    
    XY_kitsune_train = np.loadtxt(directory_data + kitsune + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_kitsune = XY_kitsune_train[:, metrics[metric]]
    X_train_kitsune = XY_kitsune_train[:, :-2]
    XY_kitsune_val = np.loadtxt(directory_data + kitsune + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_kitsune = XY_kitsune_val[:, metrics[metric]]
    X_val_kitsune = XY_kitsune_val[:, :-2]
    XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_kitsune = XY_kitsune_test[:, metrics[metric]]
    X_test_kitsune = XY_kitsune_test[:, :-2]
    
    XY_ciciot_train = np.loadtxt(directory_data + ciciot + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_ciciot = XY_ciciot_train[:, metrics[metric]]
    X_train_ciciot = XY_ciciot_train[:, :-2]
    XY_ciciot_val = np.loadtxt(directory_data + ciciot + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_ciciot = XY_ciciot_val[:, metrics[metric]]
    X_val_ciciot = XY_ciciot_val[:, :-2]
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_ciciot = XY_ciciot_test[:, metrics[metric]]
    X_test_ciciot = XY_ciciot_test[:, :-2]
    
    yi_train_kdd = pd.DataFrame(data={'ensemble': y_train_kdd, 'inference': y_pred_train_kdd})
    yi_val_kdd = pd.DataFrame(data={'ensemble': y_val_kdd, 'inference': y_pred_val_kdd})
    yi_test_kdd = pd.DataFrame(data={'ensemble': y_test_kdd, 'inference': y_pred_test_kdd})
    
    yi_train_nsl = pd.DataFrame(data={'ensemble': y_train_nsl, 'inference': y_pred_train_nsl})
    yi_val_nsl = pd.DataFrame(data={'ensemble': y_val_nsl, 'inference': y_pred_val_nsl})
    yi_test_nsl = pd.DataFrame(data={'ensemble': y_test_nsl, 'inference': y_pred_test_nsl})
    
    yi_train_ids = pd.DataFrame(data={'ensemble': y_train_ids, 'inference': y_pred_train_ids})
    yi_val_ids = pd.DataFrame(data={'ensemble': y_val_ids, 'inference': y_pred_val_ids})
    yi_test_ids = pd.DataFrame(data={'ensemble': y_test_ids, 'inference': y_pred_test_ids})
    
    yi_train_kitsune = pd.DataFrame(data={'ensemble': y_train_kitsune, 'inference': y_pred_train_kitsune})
    yi_val_kitsune = pd.DataFrame(data={'ensemble': y_val_kitsune, 'inference': y_pred_val_kitsune})
    yi_test_kitsune = pd.DataFrame(data={'ensemble': y_test_kitsune, 'inference': y_pred_test_kitsune})
    
    yi_train_ciciot = pd.DataFrame(data={'ensemble': y_train_ciciot, 'prediction': y_pred_train_ciciot})
    yi_val_ciciot = pd.DataFrame(data={'ensemble': y_val_ciciot, 'inference': y_pred_val_ciciot})
    yi_test_ciciot = pd.DataFrame(data={'ensemble': y_test_ciciot, 'inference': y_pred_test_ciciot})
    
    # mse
    print("mse kdd", mse(y_train_kdd, y_pred_train_kdd), mse(y_val_kdd, y_pred_val_kdd), mse(y_test_kdd, y_pred_test_kdd))
    print("mse nsl", mse(y_train_nsl, y_pred_train_nsl), mse(y_val_nsl, y_pred_val_nsl), mse(y_test_nsl, y_pred_test_nsl))
    print("mse ids", mse(y_train_ids, y_pred_train_ids), mse(y_val_ids, y_pred_val_ids), mse(y_test_ids, y_pred_test_ids))
    print("mse kitsune", mse(y_train_kitsune, y_pred_train_kitsune), mse(y_val_kitsune, y_pred_val_kitsune), mse(y_test_kitsune, y_pred_test_kitsune))
    print("mse ciciot", mse(y_train_ciciot, y_pred_train_ciciot), mse(y_val_ciciot, y_pred_val_ciciot), mse(y_test_ciciot, y_pred_test_ciciot))
    
    # plots
    fig, axes = plt.subplots(5, 3, figsize=(9, 9), sharey=True)
    #kdd
    sns.scatterplot(ax=axes[0, 0], data=yi_train_kdd, x="ensemble", y="inference", color='blue')
    axes[0, 0].set_title("train")
    sns.scatterplot(ax=axes[0, 1], data=yi_val_kdd, x="ensemble", y="inference", color='green')
    axes[0, 1].set_title("val")
    sns.scatterplot(ax=axes[0, 2], data=yi_test_kdd, x="ensemble", y="inference", color='red')
    axes[0, 2].set_title("test")
    #nsl
    sns.scatterplot(ax=axes[1, 0], data=yi_train_nsl, x="ensemble", y="inference", color='blue')
    sns.scatterplot(ax=axes[1, 1], data=yi_val_nsl, x="ensemble", y="inference", color='green')
    sns.scatterplot(ax=axes[1, 2], data=yi_test_nsl, x="ensemble", y="inference", color='red')
    #ids
    sns.scatterplot(ax=axes[2, 0], data=yi_train_ids, x="ensemble", y="inference", color='blue')
    sns.scatterplot(ax=axes[2, 1], data=yi_val_ids, x="ensemble", y="inference", color='green')
    sns.scatterplot(ax=axes[2, 2], data=yi_test_ids, x="ensemble", y="inference", color='red')
    #kitsune
    sns.scatterplot(ax=axes[3, 0], data=yi_train_kitsune, x="ensemble", y="inference", color='blue')
    sns.scatterplot(ax=axes[3, 1], data=yi_val_kitsune, x="ensemble", y="inference", color='green')
    sns.scatterplot(ax=axes[3, 2], data=yi_test_kitsune, x="ensemble", y="inference", color='red')
    #ciciot
    sns.scatterplot(ax=axes[4, 0], data=yi_train_ciciot, x="ensemble", y="inference", color='blue')
    sns.scatterplot(ax=axes[4, 1], data=yi_val_ciciot, x="ensemble", y="inference", color='green')
    sns.scatterplot(ax=axes[4, 2], data=yi_test_ciciot, x="ensemble", y="inference", color='red')
    plt.savefig(filename+".png", dpi=300)  
    
    
    # Pearson’s Correlation
    print("pcc kdd", pcc(y_train_kdd, y_pred_train_kdd)[0], pcc(y_val_kdd, y_pred_val_kdd)[0], pcc(y_test_kdd, y_pred_test_kdd)[0])
    print("pcc nsl", pcc(y_train_nsl, y_pred_train_nsl)[0], pcc(y_val_nsl, y_pred_val_nsl)[0], pcc(y_test_nsl, y_pred_test_nsl)[0])
    print("pcc ids", pcc(y_train_ids, y_pred_train_ids)[0], pcc(y_val_ids, y_pred_val_ids)[0], pcc(y_test_ids, y_pred_test_ids)[0])
    print("pcc kitsune", pcc(y_train_kitsune, y_pred_train_kitsune)[0], pcc(y_val_kitsune, y_pred_val_kitsune)[0], pcc(y_test_kitsune, y_pred_test_kitsune)[0])
    print("pcc ciciot", pcc(y_train_ciciot, y_pred_train_ciciot)[0], pcc(y_val_ciciot, y_pred_val_ciciot)[0], pcc(y_test_ciciot, y_pred_test_ciciot)[0])
    
    # Spearman’s Correlation
    print("scc kdd", scc(y_train_kdd, y_pred_train_kdd)[0], scc(y_val_kdd, y_pred_val_kdd)[0], scc(y_test_kdd, y_pred_test_kdd)[0])
    print("scc nsl", scc(y_train_nsl, y_pred_train_nsl)[0], scc(y_val_nsl, y_pred_val_nsl)[0], scc(y_test_nsl, y_pred_test_nsl)[0])
    print("scc ids", scc(y_train_ids, y_pred_train_ids)[0], scc(y_val_ids, y_pred_val_ids)[0], scc(y_test_ids, y_pred_test_ids)[0])
    print("scc kitsune", scc(y_train_kitsune, y_pred_train_kitsune)[0], scc(y_val_kitsune, y_pred_val_kitsune)[0], scc(y_test_kitsune, y_pred_test_kitsune)[0])
    print("scc ciciot", scc(y_train_ciciot, y_pred_train_ciciot)[0], scc(y_val_ciciot, y_pred_val_ciciot)[0], scc(y_test_ciciot, y_pred_test_ciciot)[0])

def get_lines(i, metric, filename):
    
    y_pred_train_ids = np.loadtxt(directory_outputs + ids + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_ids = np.loadtxt(directory_outputs + ids + "_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_ids = np.loadtxt(directory_outputs + ids + "_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_nsl = np.loadtxt(directory_outputs + nsl + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_nsl = np.loadtxt(directory_outputs + nsl + "_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_nsl = np.loadtxt(directory_outputs + nsl +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_kdd = np.loadtxt(directory_outputs + kdd + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_kdd = np.loadtxt(directory_outputs + kdd +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_kdd = np.loadtxt(directory_outputs + kdd +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_kitsune = np.loadtxt(directory_outputs + kitsune + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_kitsune = np.loadtxt(directory_outputs + kitsune +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_kitsune = np.loadtxt(directory_outputs + kitsune +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    y_pred_train_ciciot = np.loadtxt(directory_outputs + ciciot + "_pred_train_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_val_ciciot = np.loadtxt(directory_outputs + ciciot +"_pred_val_"+metric+"_"+str(i)+".csv", delimiter=",")
    y_pred_test_ciciot = np.loadtxt(directory_outputs + ciciot +"_pred_test_"+metric+"_"+str(i)+".csv", delimiter=",")
    
    
    XY_kdd_train = np.loadtxt(directory_data + kdd + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_kdd = XY_kdd_train[:, metrics[metric]]
    X_train_kdd = XY_kdd_train[:, :-2]
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_kdd = XY_kdd_val[:, metrics[metric]]
    X_val_kdd = XY_kdd_val[:, :-2]
    XY_kdd_test = np.loadtxt(directory_data + kdd +"_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_kdd = XY_kdd_test[:, metrics[metric]]
    X_test_kdd = XY_kdd_test[:, :-2]
        
    XY_nsl_train = np.loadtxt(directory_data + nsl + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_nsl = XY_nsl_train[:, metrics[metric]]
    X_train_nsl = XY_nsl_train[:, :-2]
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_nsl = XY_nsl_val[:, metrics[metric]]
    X_val_nsl = XY_nsl_val[:, :-2]
    XY_nsl_test = np.loadtxt(directory_data + nsl + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_nsl = XY_nsl_test[:, metrics[metric]]
    X_test_nsl = XY_nsl_test[:, :-2]
        
    XY_ids_train = np.loadtxt(directory_data + ids + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_ids = XY_ids_train[:, metrics[metric]]
    X_train_ids = XY_ids_train[:, :-2]
    XY_ids_val = np.loadtxt(directory_data + ids + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_ids = XY_ids_val[:, metrics[metric]]
    X_val_ids = XY_ids_val[:, :-2]
    XY_ids_test = np.loadtxt(directory_data + ids + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_ids = XY_ids_test[:, metrics[metric]]
    X_test_ids = XY_ids_test[:, :-2]
    
    XY_kitsune_train = np.loadtxt(directory_data + kitsune + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_kitsune = XY_kitsune_train[:, metrics[metric]]
    X_train_kitsune = XY_kitsune_train[:, :-2]
    XY_kitsune_val = np.loadtxt(directory_data + kitsune + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_kitsune = XY_kitsune_val[:, metrics[metric]]
    X_val_kitsune = XY_kitsune_val[:, :-2]
    XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_kitsune = XY_kitsune_test[:, metrics[metric]]
    X_test_kitsune = XY_kitsune_test[:, :-2]
    
    XY_ciciot_train = np.loadtxt(directory_data + ciciot + "_train_latent_" + str(i) +".csv", delimiter=',')
    y_train_ciciot = XY_ciciot_train[:, metrics[metric]]
    X_train_ciciot = XY_ciciot_train[:, :-2]
    XY_ciciot_val = np.loadtxt(directory_data + ciciot + "_val_latent_" + str(i) +".csv", delimiter=',')
    y_val_ciciot = XY_ciciot_val[:, metrics[metric]]
    X_val_ciciot = XY_ciciot_val[:, :-2]
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_ciciot = XY_ciciot_test[:, metrics[metric]]
    X_test_ciciot = XY_ciciot_test[:, :-2]
    
    yi_train_kdd = pd.DataFrame(data={'ensemble': y_train_kdd, 'inference': y_pred_train_kdd})
    yi_val_kdd = pd.DataFrame(data={'ensemble': y_val_kdd, 'inference': y_pred_val_kdd})
    yi_test_kdd = pd.DataFrame(data={'ensemble': y_test_kdd, 'inference': y_pred_test_kdd})
    
    yi_train_nsl = pd.DataFrame(data={'ensemble': y_train_nsl, 'inference': y_pred_train_nsl})
    yi_val_nsl = pd.DataFrame(data={'ensemble': y_val_nsl, 'inference': y_pred_val_nsl})
    yi_test_nsl = pd.DataFrame(data={'ensemble': y_test_nsl, 'inference': y_pred_test_nsl})
    
    yi_train_ids = pd.DataFrame(data={'ensemble': y_train_ids, 'inference': y_pred_train_ids})
    yi_val_ids = pd.DataFrame(data={'ensemble': y_val_ids, 'inference': y_pred_val_ids})
    yi_test_ids = pd.DataFrame(data={'ensemble': y_test_ids, 'inference': y_pred_test_ids})
    
    yi_train_kitsune = pd.DataFrame(data={'ensemble': y_train_kitsune, 'inference': y_pred_train_kitsune})
    yi_val_kitsune = pd.DataFrame(data={'ensemble': y_val_kitsune, 'inference': y_pred_val_kitsune})
    yi_test_kitsune = pd.DataFrame(data={'ensemble': y_test_kitsune, 'inference': y_pred_test_kitsune})
    
    yi_train_ciciot = pd.DataFrame(data={'ensemble': y_train_ciciot, 'inference': y_pred_train_ciciot})
    yi_val_ciciot = pd.DataFrame(data={'ensemble': y_val_ciciot, 'inference': y_pred_val_ciciot})
    yi_test_ciciot = pd.DataFrame(data={'ensemble': y_test_ciciot, 'inference': y_pred_test_ciciot})
    
    # plots
    y_kdd = np.concatenate((y_train_kdd, y_val_kdd, y_test_kdd), axis=0)
    y_nsl = np.concatenate((y_train_nsl, y_val_nsl, y_test_nsl), axis=0)
    y_ids = np.concatenate((y_train_ids, y_val_ids, y_test_ids), axis=0)
    y_kitsune = np.concatenate((y_train_kitsune, y_val_kitsune, y_test_kitsune), axis=0)
    y_ciciot = np.concatenate((y_train_ciciot, y_val_ciciot, y_test_ciciot), axis=0)
    
    y_pred_kdd = np.concatenate((y_pred_train_kdd, y_pred_val_kdd, y_pred_test_kdd), axis=0)
    y_pred_nsl = np.concatenate((y_pred_train_nsl, y_pred_val_nsl, y_pred_test_nsl), axis=0)
    y_pred_ids = np.concatenate((y_pred_train_ids, y_pred_val_ids, y_pred_test_ids), axis=0)
    y_pred_kitsune = np.concatenate((y_pred_train_kitsune, y_pred_val_kitsune, y_pred_test_kitsune), axis=0)
    y_pred_ciciot = np.concatenate((y_pred_train_ciciot, y_pred_val_ciciot, y_pred_test_ciciot), axis=0)
    
    
    yi_kdd = pd.DataFrame(data={'ensemble': y_kdd, 'inference': y_pred_kdd})
    yi_nsl = pd.DataFrame(data={'ensemble': y_nsl, 'inference': y_pred_nsl})
    yi_ids = pd.DataFrame(data={'ensemble': y_ids, 'inference': y_pred_ids})
    yi_kitsune = pd.DataFrame(data={'ensemble': y_kitsune, 'inference': y_pred_kitsune})
    yi_ciciot = pd.DataFrame(data={'ensemble': y_ciciot, 'inference': y_pred_ciciot})
    # lines
    fig, axes = plt.subplots(5, 1, figsize=(9, 9), sharey=False)
    n = 200
    yi_kdd = yi_kdd.sample(n = n)
    yi_nsl = yi_nsl.sample(n = n)
    yi_ids = yi_ids.sample(n = n)
    yi_kitsune = yi_kitsune.sample(n = n)
    yi_ciciot = yi_ciciot.sample(n = n)
    sns.scatterplot(ax=axes[0], x=yi_kdd.index, y='ensemble', data=yi_kdd, color='blue')
    sns.lineplot(ax=axes[0], x=yi_kdd.index, y='inference', data=yi_kdd, color='red')
    sns.scatterplot(ax=axes[1], x=yi_nsl.index, y='ensemble', data=yi_nsl, color='blue')
    sns.lineplot(ax=axes[1], x=yi_nsl.index, y='inference', data=yi_nsl, color='red')
    sns.scatterplot(ax=axes[2], x=yi_ids.index, y='ensemble', data=yi_ids, color='blue')
    sns.lineplot(ax=axes[2], x=yi_ids.index, y='inference', data=yi_ids, color='red')
    sns.scatterplot(ax=axes[3], x=yi_kitsune.index, y='ensemble', data=yi_kitsune, color='blue')
    sns.lineplot(ax=axes[3], x=yi_kitsune.index, y='inference', data=yi_kitsune, color='red')
    sns.scatterplot(ax=axes[4], x=yi_ciciot.index, y='ensemble', data=yi_ciciot, color='blue')
    sns.lineplot(ax=axes[4], x=yi_ciciot.index, y='inference', data=yi_ciciot, color='red')
    axes[0].set(xlabel=None, ylabel=None)
    axes[1].set(xlabel=None, ylabel=None)
    axes[2].set(xlabel=None, ylabel=None)
    axes[3].set(xlabel=None, ylabel=None)
    axes[4].set(xlabel=None, ylabel=None)
    axes[0].legend(labels=["ensemble","inference"])
    plt.savefig(filename+"_lines.png", dpi=300)  
    
    
def get_data_description(size, metric, filename):
    
    # plots
    fig, axes = plt.subplots(size, 5, figsize=(9, 12), sharey=True)
    pca = PCA(n_components=1)
    fig.text(0.5, 0.07, 'pca component', ha='center')
    fig.text(0.04, 0.5, 'uncertainty', va='center', rotation='vertical')
    
    for i in range(size):

        XY_kdd_train = np.loadtxt(directory_data + kdd + "_train_latent_" + str(i) +".csv", delimiter=',')
        y_train_kdd = XY_kdd_train[:, metrics[metric]]
        X_train_kdd = XY_kdd_train[:, :-2]
        XY_kdd_val = np.loadtxt(directory_data + kdd + "_val_latent_" + str(i) +".csv", delimiter=',')
        y_val_kdd = XY_kdd_val[:, metrics[metric]]
        X_val_kdd = XY_kdd_val[:, :-2]
        XY_kdd_test = np.loadtxt(directory_data + kdd +"_test_latent_" + str(i) +".csv", delimiter=',')
        y_test_kdd = XY_kdd_test[:, metrics[metric]]
        X_test_kdd = XY_kdd_test[:, :-2]
            
        XY_nsl_train = np.loadtxt(directory_data + nsl + "_train_latent_" + str(i) +".csv", delimiter=',')
        y_train_nsl = XY_nsl_train[:, metrics[metric]]
        X_train_nsl = XY_nsl_train[:, :-2]
        XY_nsl_val = np.loadtxt(directory_data + nsl + "_val_latent_" + str(i) +".csv", delimiter=',')
        y_val_nsl = XY_nsl_val[:, metrics[metric]]
        X_val_nsl = XY_nsl_val[:, :-2]
        XY_nsl_test = np.loadtxt(directory_data + nsl + "_test_latent_" + str(i) +".csv", delimiter=',')
        y_test_nsl = XY_nsl_test[:, metrics[metric]]
        X_test_nsl = XY_nsl_test[:, :-2]
            
        XY_ids_train = np.loadtxt(directory_data + ids + "_train_latent_" + str(i) +".csv", delimiter=',')
        y_train_ids = XY_ids_train[:, metrics[metric]]
        X_train_ids = XY_ids_train[:, :-2]
        XY_ids_val = np.loadtxt(directory_data + ids + "_val_latent_" + str(i) +".csv", delimiter=',')
        y_val_ids = XY_ids_val[:, metrics[metric]]
        X_val_ids = XY_ids_val[:, :-2]
        XY_ids_test = np.loadtxt(directory_data + ids + "_test_latent_" + str(i) +".csv", delimiter=',')
        y_test_ids = XY_ids_test[:, metrics[metric]]
        X_test_ids = XY_ids_test[:, :-2]
        
        XY_kitsune_train = np.loadtxt(directory_data + kitsune + "_train_latent_" + str(i) +".csv", delimiter=',')
        y_train_kitsune = XY_kitsune_train[:, metrics[metric]]
        X_train_kitsune = XY_kitsune_train[:, :-2]
        XY_kitsune_val = np.loadtxt(directory_data + kitsune + "_val_latent_" + str(i) +".csv", delimiter=',')
        y_val_kitsune = XY_kitsune_val[:, metrics[metric]]
        X_val_kitsune = XY_kitsune_val[:, :-2]
        XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test_latent_" + str(i) +".csv", delimiter=',')
        y_test_kitsune = XY_kitsune_test[:, metrics[metric]]
        X_test_kitsune = XY_kitsune_test[:, :-2]
        
        XY_ciciot_train = np.loadtxt(directory_data + ciciot + "_train_latent_" + str(i) +".csv", delimiter=',')
        y_train_ciciot = XY_ciciot_train[:, metrics[metric]]
        X_train_ciciot = XY_ciciot_train[:, :-2]
        XY_ciciot_val = np.loadtxt(directory_data + ciciot + "_val_latent_" + str(i) +".csv", delimiter=',')
        y_val_ciciot = XY_ciciot_val[:, metrics[metric]]
        X_val_ciciot = XY_ciciot_val[:, :-2]
        XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test_latent_" + str(i) +".csv", delimiter=',')
        y_test_ciciot = XY_ciciot_test[:, metrics[metric]]
        X_test_ciciot = XY_ciciot_test[:, :-2]
        
        X_kdd = np.concatenate((X_train_kdd, X_val_kdd, X_test_kdd), axis=0)
        X_nsl = np.concatenate((X_train_nsl, X_val_nsl, X_test_nsl), axis=0)
        X_ids = np.concatenate((X_train_ids, X_val_ids, X_test_ids), axis=0)
        X_kitsune = np.concatenate((X_train_kitsune, X_val_kitsune, X_test_kitsune), axis=0)
        X_ciciot = np.concatenate((X_train_ciciot, X_val_ciciot, X_test_ciciot), axis=0)
        
        y_kdd = np.concatenate((y_train_kdd, y_val_kdd, y_test_kdd), axis=0)
        y_nsl = np.concatenate((y_train_nsl, y_val_nsl, y_test_nsl), axis=0)
        y_ids = np.concatenate((y_train_ids, y_val_ids, y_test_ids), axis=0)
        y_kitsune = np.concatenate((y_train_kitsune, y_val_kitsune, y_test_kitsune), axis=0)
        y_ciciot = np.concatenate((y_train_ciciot, y_val_ciciot, y_test_ciciot), axis=0)
        
        
        X1_kdd = pca.fit_transform(X_kdd).reshape((1, -1))[0]
        X1_nsl = pca.fit_transform(X_nsl).reshape((1, -1))[0]
        X1_ids = pca.fit_transform(X_ids).reshape((1, -1))[0]
        X1_kitsune = pca.fit_transform(X_kitsune).reshape((1, -1))[0]
        X1_ciciot = pca.fit_transform(X_ciciot).reshape((1, -1))[0]
        yi_kdd_pca = pd.DataFrame(data={'pca component': X1_kdd, 'uncertainty': y_kdd})
        yi_nsl_pca = pd.DataFrame(data={'pca component': X1_nsl, 'uncertainty': y_nsl})
        yi_ids_pca = pd.DataFrame(data={'pca component': X1_ids, 'uncertainty': y_ids})
        yi_kitsune_pca = pd.DataFrame(data={'pca component': X1_kitsune, 'uncertainty': y_kitsune})
        yi_ciciot_pca = pd.DataFrame(data={'pca component': X1_ciciot, 'uncertainty': y_ciciot})
        
        
        sns.scatterplot(ax=axes[i, 0], data=yi_kdd_pca, x="pca component", y="uncertainty", color='blue')
        sns.scatterplot(ax=axes[i, 1], data=yi_nsl_pca, x="pca component", y="uncertainty", color='blue')
        sns.scatterplot(ax=axes[i, 2], data=yi_ids_pca, x="pca component", y="uncertainty", color='blue')
        sns.scatterplot(ax=axes[i, 3], data=yi_kitsune_pca, x="pca component", y="uncertainty", color='blue')
        sns.scatterplot(ax=axes[i, 4], data=yi_ciciot_pca, x="pca component", y="uncertainty", color='blue')
        axes[i, 0].set(xlabel=None, ylabel=None)
        axes[i, 1].set(xlabel=None, ylabel=None)
        axes[i, 2].set(xlabel=None, ylabel=None)
        axes[i, 3].set(xlabel=None, ylabel=None)
        axes[i, 4].set(xlabel=None, ylabel=None)
        if i==0:
            axes[i, 0].set_title("kdd")
            axes[i, 1].set_title("nsl")
            axes[i, 2].set_title("ids")
            axes[i, 3].set_title("kitsune")
            axes[i, 4].set_title("ciciot")
        print("preping plot for candidate "+str(i)+" done")
    
    plt.savefig(filename+"_description_" + metric + ".png", dpi=300)  