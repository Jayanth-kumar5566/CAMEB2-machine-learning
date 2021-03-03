from skbio.stats.composition import clr
import pandas as pd
def transformation(df_sel,x):
	'''
	x is the transformation
	clr  : Centered Log ratio
	rel  : relative abundance
	none : No transformation 
	'''
	if x == "clr":
		df_norm=clr(df_sel+1)
		df_norm=pd.DataFrame(df_norm,index=df_sel.index,columns=df_sel.columns)
	elif x == "rel":
		df_norm=df_sel.div(df_sel.sum(axis=1),axis=0)*100
	elif x == "none":
		df_norm=df_sel
	else:
		sys.exit("Invalid input parameter - transformation")
	return(df_norm)	

def data_sel(code,trans):
	y1=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_194.csv",index_col=0)
	y2=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/METADATA/data_test.csv",index_col=0)
	pat_index=pd.concat([y1,y2],axis=0).index

	if code == "I":
		df=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/I.csv",index_col=0)
		f=df.columns
	elif code == "II":
		df=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/Data_21Dec20/species_data.csv",index_col=0)		
		# Feature-selection LEFSe
		f=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/II.csv",index_col=0).index

	elif code == "III":
		df=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/MASTER-TABLES/HUMANN2/humann2_unipathway_pathabundance_cpm.tsv",sep='\t',index_col=0)
		pathways=[i.find("|")==-1 for i in df.index]
		df=df.loc[pathways,:]
		del pathways
		df.columns=[i.split("_")[0] for i in df.columns]
		df.drop(["13LTBlank","76LTBlank","Blank"],axis=1,inplace=True)
		df.drop(["UNMAPPED","UNINTEGRATED"],axis=0,inplace=True)
		df["Super_pathway"]=[i.split(";")[0] for i in df.index]
		df=df.groupby("Super_pathway")
		df=df.sum()
		df=df.transpose()
		# Feature-selection LEFSe
		f=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/III.csv",index_col=0).index
	elif code == "IV":
		df=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/MASTER-TABLES/AMR/shortbred-CARD-95-summary.csv",index_col=0)
		df=df.groupby("Drug.ClassAMR")
		df=df.sum()
		df.drop(["Blank"],axis=1,inplace=True)
		df=df.transpose()
		# Feature-selection LEFSe
		f=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/IV.csv",index_col=0).index
	elif code == "V":
		df=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/MASTER-TABLES/VIRFINDER/c10k_abundance_individual_contigs.csv",index_col=0)
		df.drop(["Blank","13LTBlank","76LTBlank"],axis=1,inplace=True)
		df=df.transpose()		
		f=pd.read_csv("/home/jayanth/OneDrive/21.ML_Bronch/Data/CAMEB2-machine-learning/Results/Feature_Selection/V.csv",index_col=0).index

	else:
		sys.exit('Input Invalid -- dataset_code')
	
	df_sel=df.loc[pat_index,f]			
	df_norm=transformation(df_sel,trans)
	return(df_norm)
