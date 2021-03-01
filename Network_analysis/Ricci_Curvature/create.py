import matplotlib.pyplot as plt
import pandas
import networkx as nx
import pickle
import OllivierRicci as OR
from joblib import Parallel, delayed
data=pandas.read_csv("./../edge_weights-across_patients_unfiltered.csv",index_col=0)

def ricci_curvature(G):
    '''G is a networkx graph with weights in the attribute weight'''
    try: 
        'weight' in list(G.edges(data=True))[0][2].keys()
        print("inside")
        #G2=OR. _compute_ricci_curvature(G,shortest_path="pairwise")
        G2=OR. _compute_ricci_curvature(G,method="Sinkhorn",nbr_topk=1000)
        return(G2)
    except:
        print("No weight data")
        return(None)

'''
def network_measures(G):
	deg=G.degree() #Degree for each node
	clus=nx.clustering(G) #Clustering coefficient for each node
	avg_clus=nx.average_clustering(G) #average clustering coefficient of the graph
	bet_cen= nx.betweenness_centrality(G) #betweeness centrality
	clo_cen= nx.closeness_centrality(G) #closness centrality
	eig_cen= nx.eigenvector_centrality(G) #eigenvector centrality	
	deg_cen= nx.degree_centrality(G) #degree centrality
	kat_cen=nx.katz_centrality(G) #kratz centrality
	ld_cen=nx.load_centrality(G) #load centrality
'''


from inspect import getmembers, isfunction
all_fun=getmembers(nx.algorithms,isfunction)

'''
res=dict()
#parallelize
for i in all_fun:
	try:
		print(i[0])	
		res[str(i[0])]=i[1](G)
		print("Applicable")
	except:
		print("Not applicable",i[0])
		pass	
'''

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
	G=nx.reverse(G)
	print(pat)
	dfs_metric[pat]=Parallel(n_jobs=16,prefer="processes")(delayed(i)(G) for i in functions)
	'''
	_Gk=OR._compute_ricci_curvature_edges(G)
	import networkit as nk
	import numpy

	apsp=nk.algebraic.adjacencyMatrix(_Gk)
	apsp=apsp.toarray()
	numpy.fill_diagonal(apsp,0)
	apsp=apsp.tolist()

	apsp=nk.distance.APSP(_Gk).run().getDistances()
	import itertools as it
	al=set([i for i in it.product(range(1,76),range(1,76))])
	ed=set(_Gk.edges())
	for (r,c) in (al-ed):
		apsp[r][c]=0
	'''

	G_n=ricci_curvature(G)
	elist_rc=nx.to_pandas_edgelist(G_n)
	f_df=pandas.DataFrame({"names":(elist_rc['source']+"_"+elist_rc['target']).values,"rc":elist_rc['ricciCurvature'].values})
	dfs[pat]=f_df
    
pickle.dump(dfs, open( "dfs.p", "wb" ) )


ricci_dataframe=pandas.DataFrame(columns=dfs.keys(),index=dfs["IT226"]["names"])

for i in dfs.keys():
    x=dfs[i].set_index("names")
    x.columns=[i]
    ricci_dataframe.update(x)

ricci_dataframe=ricci_dataframe.transpose()
ricci_dataframe.to_csv("ricci_curvature_across-patiens.csv") 

dfs_metric=pandas.DataFrame(dfs_metric,index=single_metric_list)
#dfs_metric.replace({True:0,False:1},inplace=True)
dfs_metric.to_csv("network_metrics.csv",sep='\t')
#replace true false in excel
