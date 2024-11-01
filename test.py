#to access files and folders
import os
from datetime import datetime
#data analysis and manipulation library
import pandas as pd
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

import sys
sys.path.append('/kaggle/working/cogload/')
from processing_data import Preprocessing
from selection_feature import Feature_Selection
from EDA import EDA
from best_model import train_model

from sklearn.svm import SVC

#argument parser
parser = ArgumentParser()
parser.add_argument("--data_folder_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--window_size", default = 1, type = int, help = "Window size for feature extraction SMA")
parser.add_argument("--normalize", default = "Standard", type = str, help = "Normalization method, Standard or MinMax")
parser.add_argument("--model_selected_feature", default = "None", type = str, help = "None, RFECV, SFS")
parser.add_argument("--k_features", default = 11, type = int, help = "k of feature selected of SFS")
parser.add_argument("--forward", default = False, type = bool, help = "True to use backward, False to use forward")
parser.add_argument("--floating", default = True, type = bool, help = "True to use sfs with floating, False with no floating")
parser.add_argument("--debug", default = 0, type = int, help="debug mode 0: no debug, 1: debug")

args = parser.parse_args()

args_dict = vars(args)
log_args = pd.DataFrame([args_dict])

directory_name = '/kaggle/working/log/'
if not os.path.exists(directory_name):
    os.makedirs(directory_name)
file_name = f'args.csv'  
log_args.to_csv(os.path.join(directory_name, file_name), index=False)

#read the data
label_df = pd.read_excel(args.data_folder_path+'labels.xlsx',index_col=0)
temp_df= pd.read_excel(args.data_folder_path+'temp.xlsx',index_col=0)
hr_df= pd.read_excel(args.data_folder_path+'hr.xlsx',index_col=0)
gsr_df = pd.read_excel(args.data_folder_path+'gsr.xlsx',index_col=0)
rr_df= pd.read_excel(args.data_folder_path+'rr.xlsx',index_col=0)
print("Data shapes:")
print('Labels',label_df.shape)
print('Temperature',temp_df.shape)
print('Heart Rate',hr_df.shape)
print('GSR',gsr_df.shape)
print('RR',rr_df.shape)

# Khởi tạo đối tượng Preprocessing
processing_data = Preprocessing(window_size = args.window_size, 
                                temp_df = temp_df, 
                                hr_df = hr_df, 
                                gsr_df = gsr_df, 
                                rr_df = rr_df,
                                label_df = label_df,
                                normalize=args.normalize)
X_train, y_train, X_test, y_test, user_train, user_test = processing_data.get_data(features_to_remove = "None")

print(X_train.shape,end="\n\n")
features = X_train.columns.tolist()  
features.append("None")

remove_features = []
temp = []
for i in range(1, 4):
    df = pd.DataFrame({
        'Features_removing': [],
        'Accuracy': [],
    })
    directory_name = f'/kaggle/working/log/remove_{i}_feature.csv'
    df.to_csv(directory_name, index=False)
    features = [feature for feature in features if feature not in remove_features]
    print(features)
    # for feature in features:
    #     X_train, y_train, X_test, y_test, user_train, user_test = processing_data.get_data(features_to_remove = [feature, *remove_features])

    #     train_model(X_train, 
    #                 y_train, 
    #                 X_test, 
    #                 y_test, 
    #                 user_train,
    #                 feature_remove=[feature, *remove_features], 
    #                 n_splits=args.GroupKFold, 
    #                 path = directory_name, 
    #                 debug = args.debug)
        
    # df = pd.read_csv(directory_name)
    # EDA.draw_LinePlot(os.path.dirname(directory_name), df.iloc[:,0].tolist(),  df.iloc[:,1].tolist(), f"ACCURACY_{i}")

    # # max_number = df['Accuracy'].max()
    # # name_max_number = df.loc[df['Accuracy'] == max_number, 'Features_removing']
    # name_max_number = df.loc[df['Accuracy'].idxmax(), 'Features_removing']
    # remove_features = name_max_number
