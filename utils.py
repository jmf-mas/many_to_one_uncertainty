from models.ae import AE
from models.mlp import MLP
from models.forest import FOREST
from models.utils import ae_train, mlp_train, forest_train
import torch
import torch.nn as nn
import numpy as np
from models.utils import estimate_optimal_threshold
from scipy.stats import norm
from scipy.stats import entropy

directory_model = "checkpoints/"
directory_data = "data/"
directory_output = "outputs/"
directory_output_ext = "outputs_ext/"
model_reg_name = "mlp_regressor"
forest_reg_name = "forest_regressor"
kdd = "kdd"
nsl = "nsl"
ids = "ids"
kitsune = "kitsune"
ciciot = "ciciot"
sample_size = 15
criterions = [nn.MSELoss()]*(sample_size + 1) + [nn.BCELoss()]
metrics = {"std":-2, "ent":-1}
candidates = [i for i in range(sample_size)]


device = 'cuda' if torch.cuda.is_available() else 'cpu'

def save_val_scores(model, criterion, config, X_val, y_val):
    val_score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X_val]
    params = estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100)
    eta = params["Thresh_star"]
    np.savetxt(directory_output + config + "_scores_val_" + model.name + ".csv", val_score)
    np.savetxt(directory_output + config + "_threshold_" + model.name + ".csv", [eta])
    
def save_test_scores(model, criterion, config, X_test, y_test, eta):
    test_score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X_test]
    y_pred = np.array(test_score) > eta
    y_pred = y_pred.astype(int)
    np.savetxt(directory_output + config + "_scores_test_" + model.name + ".csv", test_score)
    np.savetxt(directory_output + config + "_labels_test_" + model.name + ".csv", y_pred)
    
def get_scores(model, criterion, X):
    scores = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X]
    return scores

def get_entropies(model, criterion, eta, X):
    scores = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X]
    std_score = np.std(scores)
    pi = lambda x: norm.cdf(x, loc=eta, scale=std_score)
    entropies = [entropy([pi(score), 1-pi(score)]) for score in scores]
    return entropies
    


def get_latent_repr(model, criterion, X):
    rep = [model.enc(x_in.to(device)).detach().numpy() for x_in in X]
    rep = np.array(rep)
    score = [criterion(model(x_in.to(device))[0], x_in.to(device)).item() for x_in in X]
    score = np.array(score)
    latent_rep = np.concatenate((rep, score.reshape(-1, 1)), axis=1)
    return latent_rep

def save_uncertainty(model, config, X, label="train"):
    preds = [model(x_in.to(device)).detach().numpy() for x_in in X]
    np.savetxt(directory_output_ext + config + "_pred_" + label + ".csv", preds)
    
def save_forest_uncertainty(model, config, X, label="train"):
    preds = model.regressor.predict(X)
    np.savetxt(directory_output + config + "_pred_" + label + ".csv", preds)
 

def train_ae(batch_size = 32, lr = 1e-5, w_d = 1e-5, momentum = 0.9, epochs = 5):
    
    X_kdd_train = np.loadtxt(directory_data + kdd + "_train.csv", delimiter=',')
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val.csv", delimiter=',')
    X_nsl_train = np.loadtxt(directory_data + nsl + "_train.csv", delimiter=',')
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val.csv", delimiter=',')
    X_ids_train = np.loadtxt(directory_data + ids + "_train.csv", delimiter=',')
    XY_ids_val = np.loadtxt(directory_data + ids + "_val.csv", delimiter=',')
    X_kitsune_train = np.loadtxt(directory_data + kitsune + "_train.csv", delimiter=',')
    XY_kitsune_val = np.loadtxt(directory_data + kitsune + "_val.csv", delimiter=',')
    X_ciciot_train = np.loadtxt(directory_data + ciciot + "_train.csv", delimiter=',')
    XY_ciciot_val = np.loadtxt(directory_data + ciciot + "_val.csv", delimiter=',')
    
    configs = {kdd: [X_kdd_train, XY_kdd_val],
              nsl: [X_nsl_train, XY_nsl_val],
              ids: [X_ids_train, XY_ids_val],
              kitsune: [X_kitsune_train, XY_kitsune_val],
              ciciot: [X_ciciot_train, XY_ciciot_val]
              }
    
    
    for config in configs:
        print("training on "+config+" data set starts ...")
        X_train, XY_val = configs[config]
        X_val, y_val = XY_val[:, :-1], XY_val[:, -1]
        
        X_val = X_val.astype('float32')
        X_train = X_train.astype('float32')
        X_train = torch.from_numpy(X_train)
        X_val = torch.from_numpy(X_val)
        
        for single in range(10, sample_size):
            print("--training for ae_model_"+str(single)+" starts ...")
            model_name = "ae_model_"+config+"_"+str(single)
            ae_model = AE(X_train.shape[1], model_name)
            ae_train(ae_model, X_train, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
            ae_model.save()
            save_val_scores(ae_model, criterions[single], config, X_val, y_val)
            print("--training for ae_model_"+str(single)+" done")  
        print("training on "+config+" data set done")

    
def evaluate_ae():
    
    XY_kdd_test = np.loadtxt(directory_data + kdd + "_test.csv", delimiter=',')
    XY_nsl_test = np.loadtxt(directory_data + nsl + "_test.csv", delimiter=',')
    XY_ids_test = np.loadtxt(directory_data + ids + "_test.csv", delimiter=',')
    XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test.csv", delimiter=',')
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test.csv", delimiter=',')
    
    
    n_ids = [i for i in range(len(XY_ids_test))]
    selection_ids = np.random.choice(n_ids, size = 70000, replace=False)
    np.savetxt(directory_output + "_selection_ids.csv", selection_ids)
    n_kitsune = [i for i in range(len(XY_kitsune_test))]
    selection_kitsune = np.random.choice(n_kitsune, size = 100000, replace=False)
    np.savetxt(directory_output + "_selection_kitsune.csv", selection_kitsune)
    n_ciciot = [i for i in range(len(XY_ciciot_test))]
    selection_ciciot = np.random.choice(n_ciciot, size = 120000, replace=False)
    np.savetxt(directory_output + "_selection_ciciot.csv", selection_ciciot)
    configs = {kdd: XY_kdd_test, 
               nsl: XY_nsl_test,
               ids: XY_ids_test[selection_ids],
               kitsune: XY_kitsune_test[selection_kitsune],
               ciciot: XY_ciciot_test[selection_ciciot]}
    
    for config in configs:
        print("evaluating on "+config+" data set")
        XY_test = configs[config]
        X_test, y_test = XY_test[:, :-1], XY_test[:, -1]
        X_test = X_test.astype('float32')
        X_test = torch.from_numpy(X_test)
        
        for single in range(sample_size):
            print("--evaluation for ae_model_"+str(single))
            model_name = "ae_model_"+config+"_"+str(single)
            eta = np.loadtxt(directory_output + config + "_threshold_" + model_name + ".csv")
            ae_model = AE(X_test.shape[1], model_name)
            ae_model.load()
            ae_model.to(device)
            save_test_scores(ae_model, criterions[single], config, X_test, y_test, eta)
            print("--evaluation for ae_model_"+str(single)+" done")
        print("evaluating on "+config+" data set done")


def build_latent_representation():
    
    X_kdd_train = np.loadtxt(directory_data + kdd + "_train.csv", delimiter=',')
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val.csv", delimiter=',')
    X_nsl_train = np.loadtxt(directory_data + nsl + "_train.csv", delimiter=',')
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val.csv", delimiter=',')
    X_ids_train = np.loadtxt(directory_data + ids + "_train.csv", delimiter=',')
    XY_ids_val = np.loadtxt(directory_data + ids +"_val.csv", delimiter=',')
    X_kitsune_train = np.loadtxt(directory_data + kitsune + "_train.csv", delimiter=',')
    XY_kitsune_val = np.loadtxt(directory_data + kitsune +"_val.csv", delimiter=',')
    X_ciciot_train = np.loadtxt(directory_data + ciciot + "_train.csv", delimiter=',')
    XY_ciciot_val = np.loadtxt(directory_data + ciciot +"_val.csv", delimiter=',')
    
    XY_kdd_test = np.loadtxt(directory_data + kdd + "_test.csv", delimiter=',')
    XY_nsl_test = np.loadtxt(directory_data + nsl + "_test.csv", delimiter=',')
    XY_ids_test = np.loadtxt(directory_data + ids + "_test.csv", delimiter=',')
    XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test.csv", delimiter=',')
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test.csv", delimiter=',')
    
    configs_train = {kdd: X_kdd_train,
               nsl: X_nsl_train,
               ids: X_ids_train,
               kitsune: X_kitsune_train,
               ciciot: X_ciciot_train}
    
    configs_val = {kdd: XY_kdd_val,
               nsl: XY_nsl_val,
               ids: XY_ids_val,
               kitsune: XY_kitsune_val,
               ciciot: XY_ciciot_val}
    
    
    n_ids = [i for i in range(len(XY_ids_test))]
    selection_ids = np.random.choice(n_ids, size = 70000, replace=False)
    np.savetxt(directory_output + "_selection_ids.csv", selection_ids)
    n_kitsune = [i for i in range(len(XY_kitsune_test))]
    selection_kitsune = np.random.choice(n_kitsune, size = 100000, replace=False)
    np.savetxt(directory_output + "_selection_kitsune.csv", selection_kitsune)
    n_ciciot = [i for i in range(len(XY_ciciot_test))]
    selection_ciciot = np.random.choice(n_ciciot, size = 120000, replace=False)
    np.savetxt(directory_output + "_selection_ciciot.csv", selection_ciciot)
    configs_test = {kdd: XY_kdd_test,
               nsl: XY_nsl_test,
               ids: XY_ids_test[selection_ids],
               kitsune: XY_kitsune_test[selection_kitsune],
               ciciot: XY_ciciot_test[selection_ciciot]}
    
    
    
    configurations = {"train": configs_train,
                      "val": configs_val,
                      "test":configs_test}
    
    for configs in configurations:
        print("building latent representation on "+configs+" starts...")
        for config in configurations[configs]:
            print("--building "+config+" data set starts... ")
            if configs in ["test", "val"]:
                XY = configurations[configs][config]
                X, _ = XY[:, :-1], XY[:, -1]
            else:
                X = configurations[configs][config]
            
            X = X.astype('float32')
            X = torch.from_numpy(X)
                
            scores = []
            entropies = []
            for single in range(sample_size):
                print("----evaluation for ae_model_"+str(single)+" starts...")
                model_name = "ae_model_"+config+"_"+str(single)
                ae_model = AE(X.shape[1], model_name)
                eta = np.loadtxt(directory_output + config + "_threshold_" + model_name + ".csv")
                ae_model.load()
                ae_model.to(device)
                scores.append(get_scores(ae_model, criterions[single], X))
                entropies.append(get_entropies(ae_model, criterions[single], eta, X))
                print("----evaluation for ae_model_"+str(single)+" done")
            scores = np.array(scores)
            entropies = np.array(entropies)
            std_scores = np.std(scores, axis=0)
            mean_entropies = np.std(entropies, axis=0)
            # create latent representation with the randomly selected model
            for selected_model_id in candidates:
                selected_model_name = "ae_model_"+config+"_"+str(selected_model_id)
                selected_ae_model = AE(X.shape[1], selected_model_name)
                selected_ae_model.load()
                selected_ae_model.to(device)
                
                latent_rep = get_latent_repr(selected_ae_model, criterions[selected_model_id], X)
                latent_rep = np.concatenate((latent_rep, std_scores.reshape(-1, 1)), axis=1)
                latent_rep = np.concatenate((latent_rep, mean_entropies.reshape(-1, 1)), axis=1)
                np.savetxt(directory_data + config + "_" + configs + "_latent_" + str(selected_model_id) +".csv", latent_rep, delimiter=',')
            print("--building "+config+" data set done")
        print("building latent representation on "+configs+" done")
        
def get_train_data_configs(selected_model_id):
    XY_kdd_train = np.loadtxt(directory_data + kdd + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')

    
    XY_nsl_train = np.loadtxt(directory_data + nsl + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    
    XY_ids_train = np.loadtxt(directory_data + ids + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ids_val = np.loadtxt(directory_data + ids +"_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    
    XY_kitsune_train = np.loadtxt(directory_data + kitsune + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kitsune_val = np.loadtxt(directory_data + kitsune +"_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    
    XY_ciciot_train = np.loadtxt(directory_data + ciciot + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ciciot_val = np.loadtxt(directory_data + ciciot +"_val_latent_" + str(selected_model_id) +".csv", delimiter=',')

    
    configs = {kdd: [XY_kdd_train, XY_kdd_val],
              nsl: [XY_nsl_train, XY_nsl_val],
              ids: [XY_ids_train, XY_ids_val],
              kitsune: [XY_kitsune_train, XY_kitsune_val],
              ciciot: [XY_ciciot_train, XY_ciciot_val]}
    return configs
    
    
            
def train_mlp(batch_size = 32, lr = 1e-5, w_d = 1e-5, momentum = 0.9, epochs = 5):
    
    for selected_model_id in candidates:
        print("regression training on candidate "+str(selected_model_id) + " starts...")
        
        configs = get_train_data_configs(selected_model_id)
        
        for metric in metrics:
            print("--training for metric "+ metric + " starts...")
            for config in configs:
                print("----training on "+config+" data set starts...")
                XY_train, XY_val = configs[config]
                X_train, y_train = XY_train[:, :-2], XY_train[:, metrics[metric]]
                X_val, y_val = XY_val[:, :-2], XY_val[:, metrics[metric]]
                
                X_val = X_val.astype('float32')
                X_train = X_train.astype('float32')
                X_train = torch.from_numpy(X_train)
                X_val = torch.from_numpy(X_val)
                mlp_model = MLP(X_train.shape[1], model_reg_name + "_"+ config + "_" + metric + "_" + str(selected_model_id))
                mlp_train(mlp_model, X_train, y_train, X_val, y_val, l_r = lr, w_d = w_d, n_epochs = epochs, batch_size = batch_size)
                mlp_model.save()
                print("----training on "+config+" data set done")
            print("--training for metric "+ metric + " done")
        print("regression training on candidate "+str(selected_model_id) + " done")

def train_forest(max_depth=4):
    
    for selected_model_id in candidates:
        print("regression training on candidate "+str(selected_model_id) + " starts...")
        configs = get_train_data_configs(selected_model_id)
        
        
        for metric in metrics:
            print("--training for metric "+ metric + " starts...")
            for config in configs:
                print("----training on "+config+" data set starts...")
                XY_train, XY_val = configs[config]
                X_train, y_train = XY_train[:, :-2], XY_train[:, metrics[metric]]
                X_train = X_train.astype('float32')
                X_train = torch.from_numpy(X_train)
                forest_model = FOREST(max_depth, forest_reg_name + "_" + config + "_" + metric + "_" + str(selected_model_id))
                forest_train(forest_model, X_train, y_train, max_depth)
                forest_model.save()
                print("----training on "+config+" data set done")
            print("--training for metric "+ metric + " done")
        print("regression training on candidate "+str(selected_model_id) + " done")

def get_test_data_configs(selected_model_id):
    XY_kdd_train = np.loadtxt(directory_data + kdd + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kdd_val = np.loadtxt(directory_data + kdd + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kdd_test = np.loadtxt(directory_data + kdd +"_test_latent_" + str(selected_model_id) +".csv", delimiter=',')
        
    XY_nsl_train = np.loadtxt(directory_data + nsl + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_nsl_val = np.loadtxt(directory_data + nsl + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_nsl_test = np.loadtxt(directory_data + nsl + "_test_latent_" + str(selected_model_id) +".csv", delimiter=',')
        
    XY_ids_train = np.loadtxt(directory_data + ids + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ids_val = np.loadtxt(directory_data + ids + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ids_test = np.loadtxt(directory_data + ids + "_test_latent_" + str(selected_model_id) +".csv", delimiter=',')
        
    XY_kitsune_train = np.loadtxt(directory_data + kitsune + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kitsune_val = np.loadtxt(directory_data + kitsune + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_kitsune_test = np.loadtxt(directory_data + kitsune + "_test_latent_" + str(selected_model_id) +".csv", delimiter=',')
        
    XY_ciciot_train = np.loadtxt(directory_data + ciciot + "_train_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ciciot_val = np.loadtxt(directory_data + ciciot + "_val_latent_" + str(selected_model_id) +".csv", delimiter=',')
    XY_ciciot_test = np.loadtxt(directory_data + ciciot + "_test_latent_" + str(selected_model_id) +".csv", delimiter=',')
        
    configs = {kdd: [XY_kdd_train, XY_kdd_val, XY_kdd_test],
                  nsl: [XY_nsl_train, XY_nsl_val, XY_nsl_test],
                  ids: [XY_ids_train, XY_ids_val, XY_ids_test],
                  kitsune: [XY_kitsune_train, XY_kitsune_val, XY_kitsune_test],
                  ciciot: [XY_ciciot_train, XY_ciciot_val, XY_ciciot_test]} 
    return configs  
      
def evaluate_mlp():
    
    for selected_model_id in candidates:
        print("evaluating on candidate "+str(selected_model_id) + " starts...")
        configs = get_test_data_configs(selected_model_id)
        
        for metric in metrics:
            print("--evaluating for metric "+ metric + " starts...")
            for config in configs:
                print("----evaluating  on "+config+" data set starts... ")
                XY_train, XY_val, XY_test = configs[config]
                X_train = XY_train[:, :-2]
                X_val = XY_val[:, :-2]
                X_test = XY_test[:, :-2]
                X_train = X_train.astype('float32')
                X_val = X_val.astype('float32')
                X_test = X_test.astype('float32')
                X_train = torch.from_numpy(X_train)
                X_val = torch.from_numpy(X_val)
                X_test = torch.from_numpy(X_test)
               
                mlp_model = MLP(X_train.shape[1], model_reg_name + "_" + config +  "_" + metric + "_" + str(selected_model_id))
                mlp_model.load()
                mlp_model.to(device)
                
                save_uncertainty(mlp_model, config, X_train, label="train_" + metric + "_" + str(selected_model_id))
                save_uncertainty(mlp_model, config, X_val, label="val_" + metric + "_" + str(selected_model_id))
                save_uncertainty(mlp_model, config, X_test, label="test_" + metric + "_" + str(selected_model_id))
        
                print("----evaluating  on "+config+" data set done ")
            print("--evaluating for metric "+ metric + " done")
        print("evaluating on candidate "+str(selected_model_id) + " done")
        
def evaluate_forest(max_depth = 4):
    
    for selected_model_id in candidates:
        print("evaluating on candidate "+str(selected_model_id) + " starts...")
        configs = get_test_data_configs(selected_model_id)
        
        for metric in metrics:
            print("--evaluating for metric "+ metric + " starts...")
            for config in configs:
                print("----evaluating  on "+config+" data set starts... ")
                XY_train, XY_val, XY_test = configs[config]
                X_train = XY_train[:, :-2]
                X_val = XY_val[:, :-2]
                X_test = XY_test[:, :-2]
                X_train = X_train.astype('float32')
                X_val = X_val.astype('float32')
                X_test = X_test.astype('float32')
                X_train = torch.from_numpy(X_train)
                X_val = torch.from_numpy(X_val)
                X_test = torch.from_numpy(X_test)
               
                forest_model = FOREST(max_depth, forest_reg_name + "_" + config +  "_" + metric + "_" + str(selected_model_id))
                forest_model.load()
                
                save_forest_uncertainty(forest_model, config, X_train, label="train_" + metric + "_" + str(selected_model_id))
                save_forest_uncertainty(forest_model, config, X_val, label="val_" + metric + "_" + str(selected_model_id))
                save_forest_uncertainty(forest_model, config, X_test, label="test_" + metric + "_" + str(selected_model_id))
        
                print("----evaluating  on "+config+" data set done ")
            print("--evaluating for metric "+ metric + " done")
        print("evaluating on candidate "+str(selected_model_id) + " done")


