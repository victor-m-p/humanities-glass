# COGSCI23
import pandas as pd 
import numpy as np 
import networkx as nx 
import matplotlib.pyplot as plt 
import seaborn as sns 

# lets try to plot it as a crazy heat-matrix 


a = np.random.random((16, 16))
plt.imshow(a, cmap='hot', interpolation='nearest')
plt.show()


# actually plot the GMM