#to access files and folders
import os
import ast
from datetime import datetime
#data analysis and manipulation library
import pandas as pd
from argparse import ArgumentParser

import warnings
warnings.simplefilter("ignore")#ignore warnings during executiona

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.svm import SVC
from xgboost import XGBClassifier

import sys
sys.path.append('/kaggle/working/cogload/processData/')
from split_data import split_data
from selection_feature import Feature_Selection
from EDA import EDA
sys.path.append('/kaggle/working/cogload/model/')
from single_model import train_model as train_model_single


#argument parser
parser = ArgumentParser()
parser.add_argument("--data_folder_path", default = "/kaggle/input/cognitiveload/UBIcomp2020/last_30s_segments/", type = str, help = "Path to the data folder")
parser.add_argument("--GroupKFold", default = 3, type = int, help = "Slip data into k group")
parser.add_argument("--window_size", default = 1, type = int, help = "Window size for feature extraction SMA")
parser.add_argument("--normalize", default = "Standard", type = str, help = "Normalization method, Standard or MinMax")
parser.add_argument("--model_selected_feature", default = "None", type = str, help = "None, RFECV, SFS, SBS")
parser.add_argument("--k_features", default = 34, type = int, help = "k of feature selected of SFS")
parser.add_argument("--forward", default = False, type = bool, help = "True to use backward, False to use forward")
parser.add_argument("--floating", default = True, type = bool, help = "True to use sfs with floating, False with no floating")
parser.add_argument("--split", nargs='+', default=[1] , type=int, help="the split of data example 2 6 to split data into 2 and 6 to extract feature")
parser.add_argument("--debug", default = 0, type = int, help="debug mode 0: no debug, 1: debug")
parser.add_argument("--models_single", nargs='+', default=[] , type=str, help="models to train, 'LDA', 'SVM', 'RF','XGB'")
parser.add_argument("--models_mul", nargs='+', default=[] , type=str, help="models to train, 'MLP_Sklearn', 'MLP_Keras','TabNet'")

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

processing_data = split_data(window_size = args.window_size,
                            temp_df = temp_df,
                            hr_df = hr_df,
                            gsr_df = gsr_df,
                            rr_df = rr_df,
                            label_df = label_df,
                            normalize = args.normalize)
for i in range(len(args.split)):
    processing_data.split_data(split = args.split [i])
X_train, y_train, X_test, y_test, user_train, user_test = processing_data.get_data()
print(f'X_test : {X_test.shape}\n')
print(f'X_train : {X_train.shape}\n\n')

X_train.to_csv('/kaggle/working/X_train.csv', index=False)

if(args.model_selected_feature == "RFECV"):
    X_train, X_test = Feature_Selection.selected_RFECV(X_train = X_train,
                                                        X_test = X_test, 
                                                        y_train = y_train,
                                                        user_train = user_train
                                                        )
if(args.model_selected_feature == "SFS"):
    X_train, X_test = Feature_Selection.selected_SFS(X_train = X_train,
                                                     X_test = X_test, 
                                                     y_train = y_train,
                                                     model = SVC(kernel='linear'),
                                                     k_features = args.k_features, 
                                                     forward = args.forward,
                                                     floating = args.floating
                                                     )
if args.model_selected_feature == 'SBS':
    Feature_Selection = Feature_Selection.selected_SBS(X_train = X_train,
                                                   X_test = X_test, 
                                                   y_train = y_train, 
                                                   y_test = y_test, 
                                                   user_train = user_train,
                                                   models = args.models_single + args.models_mul,
                                                   features_number = args.k_features,
                                                   )
    X_train = X_train[Feature_Selection]
    X_test = X_test[Feature_Selection]

print(f'X_train : {X_train.shape}\n\n')
X_train.to_csv('/kaggle/working/X_train_Selected.csv', index=False)

# if args.models_single != []:
#     train_model_single(X_train, 
#             y_train, 
#             X_test, 
#             y_test, 
#             user_train,
#             n_splits=args.GroupKFold, 
#             path = directory_name + 'single_model/', 
#             debug = args.debug,
#             models= args.models_single)

# if args.models_mul != []:
#     from mul_model import train_model as train_model_mul
#     train_model_mul(X_train,
#                 y_train, 
#                 X_test, 
#                 y_test, 
#                 user_train,
#                 n_splits=args.GroupKFold, 
#                 path = directory_name + 'mul_model/', 
#                 debug = args.debug,
#                 models= args.models_mul)