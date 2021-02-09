import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas
import numpy as np
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from scipy.interpolate import griddata

df=pandas.read_csv("tuning_res.csv",index_col=0)
'''
fig = plt.figure()
ax = Axes3D(fig)

ax.plot_trisurf(df["n_estimators"], df["max_depth"],df["Class_average Accuracy"], cmap=cm.jet, linewidth=0.2)
plt.show()
'''

#With smoothing using cubic interpolation

# 2D-arrays from DataFrame
x1 = np.linspace(df['n_estimators'].min(), df['n_estimators'].max(), len(df['n_estimators'].unique()))
y1 = np.linspace(df['max_depth'].min(), df['max_depth'].max(), len(df['max_depth'].unique()))

x2, y2 = np.meshgrid(x1, y1)

# Interpolate unstructured D-dimensional data.
z2 = griddata((df['n_estimators'], df['max_depth']), df['Class_average Accuracy'], (x2, y2), method='cubic')

# Ready to plot
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(x2, y2, z2, rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)

#ax.set_zlim(0, 1)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel("n_estimators")
ax.set_ylabel("max_depth")
ax.set_zlabel("Mean Class/Group accuracy")

fig.colorbar(surf, shrink=0.5, aspect=5)
#plt.title('Meshgrid Created from 3 1D Arrays')

plt.show()


for xaxis in ['n_estimators', 'min_samples_leaf','max_depth', 'min_samples_split']:
	plt.scatter(df[xaxis],df["Class_average Accuracy"],color="blue",label="testing")
	plt.scatter(df[xaxis],df["training_Acc"],color="red",label="training")
	plt.xlabel(xaxis)
	plt.legend()
	plt.show()
