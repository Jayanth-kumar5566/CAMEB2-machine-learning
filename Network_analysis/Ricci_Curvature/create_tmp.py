import matplotlib.pyplot as plt
import pandas
import networkx as nx
import pickle
import OllivierRicci as OR
from joblib import Parallel, delayed
data=pandas.read_csv("./../edge_weights-across_patients_unfiltered.csv",index_col=0)

from inspect import getmembers, isfunction
all_fun=getmembers(nx.algorithms,isfunction)

single_metric_list=["average_clustering","average_shortest_path_length","degree_assortativity_coefficient","degree_pearson_correlation_coefficient","flow_hierarchy","global_reaching_centrality",
'is_aperiodic', 'is_arborescence', 'is_attracting_component', 'is_bipartite', 'is_branching','is_directed_acyclic_graph', 'is_distance_regular',  'is_eulerian', 'is_forest','is_multigraphical', 'is_pseudographical', 'is_regular', 'is_semiconnected', 'is_semieulerian', 'is_strongly_connected', 'is_strongly_regular', 'is_tree', 'is_triad','is_weakly_connected','negative_edge_cycle','number_attracting_components','number_of_isolates','number_strongly_connected_components','number_weakly_connected_components','overall_reciprocity','reciprocity','trophic_incoherence_parameter']

functions=[i[1] for i in all_fun if i[0] in single_metric_list]

	
dfs_metric={}
dfs={} #dataframe for each patient
for pat in data.index:
	elist=data.loc[pat,:]
	#elist=abs(elist) #Absolute values of the edges
	m1=[i.split("_")[0]+" "+i.split("_")[1] for i in elist.index]
	m2=[i.split("_")[2]+" "+i.split("_")[3] for i in elist.index]
	ed_list= pandas.DataFrame({"x":m1,"y":m2,"weight":elist.values})
	ed_list=ed_list.loc[ed_list["x"]!=ed_list["y"],:] #removing self loops
	G = nx.DiGraph()
	G.add_weighted_edges_from([tuple(x) for x in ed_list.values])
	#Reverse the graph ?
#	G=nx.reverse(G)
	print(pat)
	dfs_metric[pat]=Parallel(n_jobs=16,prefer="processes")(delayed(i)(G) for i in functions)
	

dfs_metric=pandas.DataFrame(dfs_metric,index=single_metric_list)
#dfs_metric.replace({True:0,False:1},inplace=True)
dfs_metric.to_csv("network_metrics_no-rev.csv",sep='\t')
#replace true false in excel
