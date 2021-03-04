#!/usr/bin/env python3

# Single pipeline to run all the analysis
import sys
import os
import pandas
from joblib import Parallel, delayed

args=sys.argv

'''
Run as:
	./pipeline_wilcoxon.py dataset alpha dimension_reduction ml_algo param_dimred param_ml

dataset:
	I   - clinical attributes
	II  - Microbiome (from Kaiju)
	III - Microbial pathways (Unipathways)
	IV  - Anti Microbial resistance (HUMMAN2)
	V   - Bacteriophages contigs (Virome)

	Can also include I+II (or such combinations)	
	
alpha: alpha value 
	0.1 (90%) or 0.05 (95%) used for Wilcoxon test - Feature selection
	
dimension_reduction	: "none", "pca", "VAE", "rp"
ml_algo 			: "logit", "rf" 
param_dimred		: parameters for dimension reduction algorthim seperated by ,
	"none"	- none
	"pca"	- ratio(explanined variance to preserve)	
	"rp"	- epsilon value
	"vae"	- dimensions 80,20
param_ml			: parameters for ML algorithm
	"none"	- none  [all built in] #just incase
'''
#Dataset

#parse dataset code
dataset_code=args[1].split("+")

#Feature selection with Wilcoxon
def dataset(number,args):
	if number=="I":
		os.system("python3 Codes_pipeline/pre-process_clinical.py")
		os.system("./Codes_pipeline/fet_sel_clinical.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/"+number+".csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv "+args[2])
		return(None)
	elif number=="II":
		os.system("python3 Codes_pipeline/pre-process.py")
		os.system("./Codes_pipeline/fet_sel.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/"+number+".csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv "+args[2])
		return(None)
	elif number=="III":
		os.system("python3 Codes_pipeline/pre-process_pathways.py")	
		os.system("./Codes_pipeline/fet_sel.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/"+number+".csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv "+args[2])
		return(None)
	elif number=="IV":
		os.system("python3 Codes_pipeline/pre-process_amr.py")
		os.system("./Codes_pipeline/fet_sel.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/"+number+".csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv "+args[2])	
		return(None)
	elif number=="V":
		os.system("python3 Codes_pipeline/pre-process_phage.py")
		os.system("./Codes_pipeline/fet_sel.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/"+number+".csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv "+args[2])		
		return(None)
	else:
		sys.exit("Please check your input")
		
#Execute Part 1 of the pipeline - dataset processing and feature selection
Parallel(n_jobs=16)(delayed(dataset)(i,args) for i in dataset_code) #Run in parallel 

#=============Part 2=========================

def import_fsel_data(number):
    df=pandas.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/"+number+".csv",index_col=0)
    return(df)
	
datasets=[import_fsel_data(i) for i in dataset_code]

#concatenate all the datafames in the list
data = pandas.concat([df.stack() for df in datasets], axis=0).unstack()
data.to_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/final_data.csv")

#=============Part 3============================
#Machine Learning part
os.system("./Codes_pipeline/machine_learning.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/final_data.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv "+args[3]+" "+args[4]+" "+args[5]+" "+args[6]+" /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Machine_Learning/")

