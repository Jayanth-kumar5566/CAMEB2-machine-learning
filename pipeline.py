#!/usr/bin/env python3

# Single pipeline to run all the analysis
import sys
import os
from joblib import Parallel, delayed

args=sys.argv
'''
Run as:
	pipeline.py dataset alpha 

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

'''
#Dataset

#parse dataset code

dataset_code=args[1].split("+")

#Feature selection with LEFSe
def dataset(number,args):
	if number=="I":
		'''Clinical attributes yet to run
		'''
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
		
datasets=Parallel(n_jobs=16)(delayed(dataset)(i,args) for i in dataset_code) #Run in parallel 


