#!/usr/bin/env python3

# Single pipeline to run all the analysis
import sys
import os
import pandas
from joblib import Parallel, delayed

args=sys.argv

'''
Run as:
	./pipeline.py dataset alpha lda_thres transformation dimension_reduction ml_algo param_dimred param_ml

dataset:
	I   - clinical attributes
	II  - Microbiome (from Kaiju)
	III - Microbial pathways (Unipathways)
	IV  - Anti Microbial resistance (HUMMAN2)
	V   - Bacteriophages contigs (Virome)

	Can also include I+II (or such combinations)	
	
alpha: alpha value 
	0.1 (90%) or 0.05 (95%) used for LEFSe and Wilcoxon test - Feature selection
	
lda_thres: absolute lda  threshold
	0 or 0.2 used for LEFSe threshold
	
transformation : The transformation to apply on the counts/raw data 	
	clr  : Centered Log ratio
	rel  : relative abundance
	none : No transformation 
	arc  : arc_sine transformation #yet to implement
	
	should include transformations for each code seperated by ',' (maintaining the order) 
	example: none,clr
	
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
trans_code=args[4].split(",")
#Feature selection with LEFSe
def dataset(number,args):
	if number=="I":
		os.system("python3 Codes_pipeline/pre-process_clinical.py")
		os.system("./Codes_pipeline/fet_sel_clinical.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/I.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/I.csv")
		return(None)
	elif number=="II":
		os.system("python3 Codes_pipeline/pre-process.py")
		os.system("./Codes_pipeline/format_input.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.in -f c -c 2 -u 1 -s -1 -o 1000000")
		os.system(str("./Codes_pipeline/run_lefse.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.in /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.res -a "+args[2]+" -w "+args[2]+" -l "+args[3]))	
		os.system(str("./Codes_pipeline/parse.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/II.res /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/II.csv"))
		os.system("./Codes_pipeline/plot.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/II.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/II.png")
		return(None)
	elif number=="III":
		os.system("python3 Codes_pipeline/pre-process_pathways.py")	
		os.system("./Codes_pipeline/format_input.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III.in -f c -c 2 -u 1 -s -1 -o 1000000")
		os.system(str("./Codes_pipeline/run_lefse.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III.in /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III.res -a "+args[2]+" -w "+args[2]+" -l "+args[3]))
		os.system(str("./Codes_pipeline/parse_pathways.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III.res /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/III.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/III_index_dict_.obj"))
		os.system("./Codes_pipeline/plot.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/III.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/III.png")
		return(None)
	elif number=="IV":
		os.system("python3 Codes_pipeline/pre-process_amr.py")
		os.system("./Codes_pipeline/format_input.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV.in -f c -c 2 -u 1 -s -1 -o 1000000")
		os.system(str("./Codes_pipeline/run_lefse.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV.in /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV.res -a "+args[2]+" -w "+args[2]+" -l "+args[3]))
		os.system(str("./Codes_pipeline/parse_pathways.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV.res /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/IV.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/IV_index_dict_.obj"))
		os.system("./Codes_pipeline/plot.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/IV.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/IV.png")		
		return(None)
	elif number=="V":
		os.system("python3 Codes_pipeline/pre-process_phage.py")
		os.system("./Codes_pipeline/format_input.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.in -f c -c 2 -u 1 -s -1 -o 1000000")
		os.system(str("./Codes_pipeline/run_lefse.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.in /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.res -a "+args[2]+" -w "+args[2]+" -l "+args[3]))
		os.system(str("./Codes_pipeline/parse_pathways.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V.res /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/V.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Datasets/V_index_dict_.obj"))
		os.system("./Codes_pipeline/plot.R /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/V.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/V.png")			
		return(None)
	else:
		sys.exit("Please check your input")
		
#Execute Part 1 of the pipeline - dataset processing and feature selection
Parallel(n_jobs=16)(delayed(dataset)(i,args) for i in dataset_code) #Run in parallel 

#=============Part 2=========================
#Exception block
if len(dataset_code)!= len(trans_code): 
	sys.exit('Please check input -- inconsistent dataset code and transformation')
	
import Codes_pipeline.part2_data_merge as dm
datasets=Parallel(n_jobs=16)(delayed(dm.data_sel)(dataset_code[i],trans_code[i]) for i in range(len(dataset_code)))	

#concatenate all the datafames in the list
data = pandas.concat([df.stack() for df in datasets], axis=0).unstack()
data.to_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/final_data.csv")
#=============Part 3============================
#Machine Learning part
os.system("./machine_learning.py /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/final_data.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv /home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv "+args[5]+" "+args[6]+" "+args[7]+" "+args[8]+" /home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Machine_Learning/")
