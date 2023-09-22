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
    y_val_ciciot = XY_ids_val[:, metrics[metric]]
    X_val_ciciot = XY_ciciot_val[:, :-2]
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test_latent_" + str(i) +".csv", delimiter=',')
    y_test_ciciot = XY_ciciot_test[:, metrics[metric]]
    X_test_ciciot = XY_ciciot_test[:, :-2]
    
    yi_train_kdd = pd.DataFrame(data={'ensemble': y_train_kdd, 'prediction': y_pred_train_kdd})
    yi_val_kdd = pd.DataFrame(data={'ensemble': y_val_kdd, 'prediction': y_pred_val_kdd})
    yi_test_kdd = pd.DataFrame(data={'ensemble': y_test_kdd, 'prediction': y_pred_test_kdd})
    
    yi_train_nsl = pd.DataFrame(data={'ensemble': y_train_nsl, 'prediction': y_pred_train_nsl})
    yi_val_nsl = pd.DataFrame(data={'ensemble': y_val_nsl, 'prediction': y_pred_val_nsl})
    yi_test_nsl = pd.DataFrame(data={'ensemble': y_test_nsl, 'prediction': y_pred_test_nsl})
    
    yi_train_ids = pd.DataFrame(data={'ensemble': y_train_ids, 'prediction': y_pred_train_ids})
    yi_val_ids = pd.DataFrame(data={'ensemble': y_val_ids, 'prediction': y_pred_val_ids})
    yi_test_ids = pd.DataFrame(data={'ensemble': y_test_ids, 'prediction': y_pred_test_ids})
    
    yi_train_kitsune = pd.DataFrame(data={'ensemble': y_train_kitsune, 'prediction': y_pred_train_kitsune})
    yi_val_kitsune = pd.DataFrame(data={'ensemble': y_val_kitsune, 'prediction': y_pred_val_kitsune})
    yi_test_kitsune = pd.DataFrame(data={'ensemble': y_test_kitsune, 'prediction': y_pred_test_kitsune})
    
    yi_train_ciciot = pd.DataFrame(data={'ensemble': y_train_ciciot, 'prediction': y_pred_train_ciciot})
    yi_val_ciciot = pd.DataFrame(data={'ensemble': y_val_ciciot, 'prediction': y_pred_val_ciciot})
    yi_test_ciciot = pd.DataFrame(data={'ensemble': y_test_ciciot, 'prediction': y_pred_test_ciciot})
    
    pca = PCA(n_components=1)
    yi_kdd_pca = pd.DataFrame(data={'pca component': pca.fit_transform(X_test_kdd), 'uncertainty': y_test_kdd})
    yi_nsl_pca = pd.DataFrame(data={'pca component': pca.fit_transform(X_test_nsl), 'uncertainty': y_test_nsl})
    yi_ids_pca = pd.DataFrame(data={'pca component': pca.fit_transform(X_test_ids), 'uncertainty': y_test_ids})
    yi_kitsune_pca = pd.DataFrame(data={'pca component': pca.fit_transform(X_test_kitsune), 'uncertainty': y_test_kitsune})
    yi_ciciot_pca = pd.DataFrame(data={'pca component': pca.fit_transform(X_test_ciciot), 'uncertainty': y_test_ciciot})

    # mse
    print("mse kdd", mse(y_train_kdd, y_pred_train_kdd), mse(y_val_kdd, y_pred_val_kdd), mse(y_test_kdd, y_pred_test_kdd))
    print("mse nsl", mse(y_train_nsl, y_pred_train_nsl), mse(y_val_nsl, y_pred_val_nsl), mse(y_test_nsl, y_pred_test_nsl))
    print("mse ids", mse(y_train_ids, y_pred_train_ids), mse(y_val_ids, y_pred_val_ids), mse(y_test_ids, y_pred_test_ids))
    print("mse kitsune", mse(y_train_kitsune, y_pred_train_kitsune), mse(y_val_kitsune, y_pred_val_kitsune), mse(y_test_kitsune, y_pred_test_kitsune))
    print("mse ciciot", mse(y_train_ciciot, y_pred_train_ciciot), mse(y_val_ciciot, y_pred_val_ciciot), mse(y_test_ciciot, y_pred_test_ciciot))
    
    # plots
    fig, axes = plt.subplots(5, 3, figsize=(9, 9), sharey=True)
    #kdd
    sns.scatterplot(ax=axes[0, 0], data=yi_train_kdd, x="ensemble", y="prediction", color='blue')
    axes[0, 0].set_title("train")
    sns.scatterplot(ax=axes[0, 1], data=yi_val_kdd, x="ensemble", y="prediction", color='green')
    axes[0, 1].set_title("val")
    sns.scatterplot(ax=axes[0, 2], data=yi_test_kdd, x="ensemble", y="prediction", color='red')
    axes[0, 2].set_title("test")
    #nsl
    sns.scatterplot(ax=axes[1, 0], data=yi_train_nsl, x="ensemble", y="prediction", color='blue')
    sns.scatterplot(ax=axes[1, 1], data=yi_val_nsl, x="ensemble", y="prediction", color='green')
    sns.scatterplot(ax=axes[1, 2], data=yi_test_nsl, x="ensemble", y="prediction", color='red')
    #ids
    sns.scatterplot(ax=axes[2, 0], data=yi_train_ids, x="ensemble", y="prediction", color='blue')
    sns.scatterplot(ax=axes[2, 1], data=yi_val_ids, x="ensemble", y="prediction", color='green')
    sns.scatterplot(ax=axes[2, 2], data=yi_test_ids, x="ensemble", y="prediction", color='red')
    #kitsune
    sns.scatterplot(ax=axes[3, 0], data=yi_train_kitsune, x="ensemble", y="prediction", color='blue')
    sns.scatterplot(ax=axes[3, 1], data=yi_val_kitsune, x="ensemble", y="prediction", color='green')
    sns.scatterplot(ax=axes[3, 2], data=yi_test_kitsune, x="ensemble", y="prediction", color='red')
    #ciciot
    sns.scatterplot(ax=axes[4, 0], data=yi_train_ciciot, x="ensemble", y="prediction", color='blue')
    sns.scatterplot(ax=axes[4, 1], data=yi_val_ciciot, x="ensemble", y="prediction", color='green')
    sns.scatterplot(ax=axes[4, 2], data=yi_test_ciciot, x="ensemble", y="prediction", color='red')
    plt.savefig(filename+".png", dpi=300)  
    
    
    fig, axes = plt.subplots(1, 5, figsize=(2, 10), sharey=True)
    sns.scatterplot(ax=axes[0, 0], data=yi_kdd_pca, x="pca component", y="uncertainty", color='blue')
    axes[0, 0].set_title("kdd")
    sns.scatterplot(ax=axes[0, 1], data=yi_nsl_pca, x="pca component", y="uncertainty", color='blue')
    axes[0, 1].set_title("nsl")
    sns.scatterplot(ax=axes[0, 2], data=yi_ids_pca, x="pca component", y="uncertainty", color='blue')
    axes[0, 2].set_title("ids")
    sns.scatterplot(ax=axes[0, 3], data=yi_kitsune_pca, x="pca component", y="uncertainty", color='blue')
    axes[0, 3].set_title("kitsune")
    sns.scatterplot(ax=axes[0, 4], data=yi_ciciot_pca, x="pca component", y="uncertainty", color='blue')
    axes[0, 4].set_title("ciciot")

    plt.savefig(filename+"_description.png", dpi=300)  
    
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