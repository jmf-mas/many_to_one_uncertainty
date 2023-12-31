from utils import train_ae, train_mlp, build_latent_representation, evaluate_mlp
from utils import train_forest, evaluate_forest
from pathlib import Path
import argparse
from data_processing import process_raw_data, split_and_save_data


directory_model = "checkpoints/"
directory_data = "data/"
directory_output = "outputs/"
directory_output_ext = "outputs_ext/"
directory_plots = "plots/"


def init():
    Path(directory_model).mkdir(parents=True, exist_ok=True)
    Path(directory_data).mkdir(parents=True, exist_ok=True)
    Path(directory_output).mkdir(parents=True, exist_ok=True)
    Path(directory_output_ext).mkdir(parents=True, exist_ok=True)
    Path(directory_plots).mkdir(parents=True, exist_ok=True)

def run_ae(batch_size, lr, w_d, momentum, epochs, is_train, to_process_data):
    
    if to_process_data:
        process_raw_data()
        split_and_save_data()
        
    if is_train:
        init()
        train_ae(batch_size, lr, w_d, momentum, epochs)
    
    build_latent_representation()
    
def run_mlp(batch_size, lr, w_d, momentum, epochs, is_train_mlp):
    
        
    if is_train_mlp:
        train_mlp(batch_size, lr, w_d, momentum, epochs)
    
    evaluate_mlp()

def run_forest(max_depth, is_train_mlp):
    
        
    if is_train_mlp:
        train_forest(max_depth)
    
    evaluate_forest()

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Regression-based UQ for OoD detection",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   parser.add_argument("-s", "--reg_stage", action="store_true", help="regression_stage", default=False)
   parser.add_argument("-f", "--random_forest_model", action="store_true", help="random_forest_regression_stage", default=False)
   parser.add_argument("-t", "--is_train", action="store_true", help="training mode", default=False)
   parser.add_argument("-r", "--is_train_mlp", action="store_true", help="mlp_training mode", default=False)
   parser.add_argument("-p", "--to_process_data", action="store_true", help="process data", default=False)
   parser.add_argument('-b', '--batch_size', nargs='?', const=1, type=int, default=32) 
   parser.add_argument('-l', '--learning_rate', nargs='?', const=1, type=float, default=1e-5) 
   parser.add_argument('-w', '--weight_decay', nargs='?', const=1, type=float, default=1e-5) 
   parser.add_argument('-m', '--momentum', nargs='?', const=1, type=float, default=0.9) 
   parser.add_argument('-e', '--epochs', nargs='?', const=1, type=int, default=20) 
   #parser.add_argument("src", help="Source location")
   args = parser.parse_args()
   configs = vars(args)
   reg_stage = configs['reg_stage']
   is_train = configs['is_train']
   is_train_mlp = configs['is_train_mlp']
   is_forest = configs['random_forest_model']
   to_process_data = configs['to_process_data']
   batch_size = configs['batch_size']
   lr = configs['learning_rate']
   w_d = configs['weight_decay']       
   momentum = configs['momentum']
   epochs = configs['epochs']
   
   if not reg_stage:
       run_ae(batch_size, lr, w_d, momentum, epochs, is_train, to_process_data)
   else:  
       
       if not is_forest:
           run_mlp(batch_size, lr, w_d, momentum, epochs, is_train_mlp) 
       else:
           run_forest(4, is_train_mlp)
    
