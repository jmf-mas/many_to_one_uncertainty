import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import pickle
from pathlib import Path
from sklearn import metrics as sk_metrics
from tqdm import tqdm
import torch.optim as optim
import copy

file_extension = ".csv"
parent_name ="checkpoints/"
output_directory ="outputs/"


class Loader(torch.utils.data.Dataset):
    def __init__(self):
        super(Loader, self).__init__()
        self.dataset = ''
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        row = self.dataset.iloc[idx]
        row = row.drop(labels={'label'})
        data = torch.from_numpy(np.array(row)).float()
        return data
    
class Train_Loader(Loader):
    def __init__(self):
        super(Train_Loader, self).__init__()
        self.dataset = pd.read_csv(
                       'data/random_train.csv',
                       index_col=False
                       )

def ae_train(model, X_loader, l_r = 1e-2, w_d = 1e-5, n_epochs = 1, batch_size = 32, save_errors = True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=l_r, weight_decay=w_d)
    criterion = nn.MSELoss(reduction='mean')
    model.train(True)
    errors = []
    for epoch in tqdm(range(n_epochs)):
        epoch_loss = []
        for step, batch in enumerate(X_loader):
            x_in = batch.type(torch.float32)
            x_in = x_in.to(device)
            x_out = model(x_in)
            loss = criterion(x_out, x_in)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            torch.autograd.set_detect_anomaly(True)
            epoch_loss.append(loss.item())
        errors.append(sum(epoch_loss)/len(epoch_loss))
        print("epoch {}: {}".format(epoch+1, sum(epoch_loss)/len(epoch_loss)))
    if save_errors:
        np.savetxt(output_directory+"_training_loss_"+model.name+file_extension, errors, delimiter=',')

def mlp_train(model, X_train, y_train, X_test, y_test, l_r = 1e-2, w_d = 1e-5, n_epochs = 1, batch_size = 32, save_errors = True):
    # training parameters
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)
    batch_start = torch.arange(0, len(X_train), batch_size)
     
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
    
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.regressor.parameters(), lr=0.0001)
     
    # training loop
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
     
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
            
def model_eval(model, x):
    loss_fn = nn.MSELoss()
    model.eval()
    x_pred = model(x)
    loss_val = loss_fn(x, x_pred)
    return loss_val

def compute_metrics(val_score, y_val, thresh, pos_label=1):
    """
    This function compute metrics for a given threshold

    Parameters
    ----------
    test_score
    y_test
    thresh
    pos_label

    Returns
    -------

    """
    y_pred = (val_score >= thresh).astype(int)
    y_true = y_val.astype(int)

    accuracy = sk_metrics.accuracy_score(y_true, y_pred)
    precision, recall, f_score, _ = sk_metrics.precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label=pos_label
    )
    avgpr = sk_metrics.average_precision_score(y_true, val_score)
    roc = sk_metrics.roc_auc_score(y_true, val_score)
    cm = sk_metrics.confusion_matrix(y_true, y_pred, labels=[1, 0])

    return accuracy, precision, recall, f_score, roc, avgpr, cm

def estimate_optimal_threshold(val_score, y_val, pos_label=1, nq=100):
    ratio = 100 * sum(y_val == 0) / len(y_val)
    print(f"Ratio of normal data:{ratio}")
    q = np.linspace(ratio - 5, min(ratio + 5, 100), nq)
    thresholds = np.percentile(val_score, q)

    result_search = []
    confusion_matrices = []
    f1 = np.zeros(shape=nq)
    r = np.zeros(shape=nq)
    p = np.zeros(shape=nq)
    auc = np.zeros(shape=nq)
    aupr = np.zeros(shape=nq)
    qis = np.zeros(shape=nq)

    for i, (thresh, qi) in enumerate(zip(thresholds, q)):
        # print(f"Threshold :{thresh:.3f}--> {qi:.3f}")
        # Prediction using the threshold value
        accuracy, precision, recall, f_score, roc, avgpr, cm = compute_metrics(val_score, y_val, thresh, pos_label)

        confusion_matrices.append(cm)
        result_search.append([accuracy, precision, recall, f_score])
        # print(f"qi:{qi:.3f} ==> p:{precision:.3f}  r:{recall:.3f}  f1:{f_score:.3f}")
        f1[i] = f_score
        r[i] = recall
        p[i] = precision
        auc[i] = roc
        aupr[i] = avgpr
        qis[i] = qi

    arm = np.argmax(f1)

    return {
        "Precision": p[arm],
        "Recall": r[arm],
        "F1-Score": f1[arm],
        "AUPR": aupr[arm],
        "AUROC": auc[arm],
        "Thresh_star": thresholds[arm],
        "Quantile_star": qis[arm]
    }
 
def save(model):
    Path(parent_name).mkdir(parents=True, exist_ok=True)
    with open(parent_name+model.name+".pickle", "wb") as fp:
        pickle.dump(model.state_dict(), fp)

def load(model):
    with open(parent_name+model.name+".pickle", "rb") as fp:
        model.load_state_dict(pickle.load(fp))
        
    
    
                     
    
    
    
        


