import math
import networkx as nx
import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import matplotlib.colors
import multiprocessing as mp
from sklearn.neighbors import kneighbors_graph
from scipy.special import gamma
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

'''
Density estimation
'''

class f_estimator:
    def __init__(self, n, points = None, distance_matrix = None, kernel = None):
        '''
        Enter either an nparray of points and a point x given as a np vector, or
        nparray distance_matrix and a point x given as an integer (index of the point).
        kernel is a kernel function (e.g. gaussK or biweight. Default is biweight)
        
        n: dimension of manifold
        '''
        assert points is not None or distance_matrix is not None, "Must enter either an array of points or distance_matrix"
        self.n = n
        self.points = points
        self.distance_matrix = distance_matrix
        if kernel is None:
            self.kernel = f_estimator.biweight
        else:
            self.kernel = kernel
        if points is not None:
            self.N = points.shape[0]
        else:
            self.N = distance_matrix.shape[0]
        self.h = f_estimator.bandwidth(self.N, n)   # Scott's rule
            
    def __call__(self, x):
        if self.points is not None:
            return sum([self.kernel(np.linalg.norm(y - x)/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for y in self.points])
        else:
            return sum([self.kernel(self.distance_matrix[x][i]/self.h, self.n)/(self.N*math.pow(self.h, self.n)) for i in range(self.N)])
        
    def gauss(x, n):
        '''
        Returns Gaussian kernel evaluated at point x
        '''
        return (1/math.pow(math.sqrt(2*math.pi), n))*math.exp(-x*x/2)
    
    def density(self):
        with mp.Pool(mp.cpu_count()) as p:
            density = p.map(self, self.points)
        return density
    
    def biweight(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 2/(n+2) + 1/(n+4))
            return ((1-x**2)**2)/normalization
        else:
            return 0
    
    def epanechnikov(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(1/n - 1/(n+2))
            return (1-x**2)/normalization
        else:
            return 0
        
    def triweight(x, n):
        if -1 < x < 1:
            s = 2*((math.pi)**(n/2))/gamma(n/2)
            normalization = s*(-1/(n+6) + 3/(n+4) - 3/(n+2) + 1/n)
            return ((1-x**2)**3)/normalization
        else:
            return 0
    
    def bandwidth(N, d):
        return N**(-1./(d+4))
    
'''
Geodesic Distance Estimation
'''
class Geodist_estimator:
    def __init__(self, X, n, k = None, ell = 6, kernel = f_estimator.biweight):
        self.X = X
        self.N = X.shape[0]
        self.n = n
        self.ell = ell
        if k is None:
            self.k = self.choose_k()
            print("Set k = ", self.k)
        else:
            self.k = k
        self.kernel = kernel
        self.Gknn = None

    def alpha(self):
        '''
        Scaling factor as number of points N --> infty, in a manifold of dimension n.
        '''
        alpha = (self.N/(math.log(self.N)*(math.log(self.N) + (self.n-1)*math.log(math.log(self.N)))))**(1.0/self.n)
        return alpha
    
    def choose_k(self):
        num_comps = [None if i > 0 else self.N for i in range(self.ell)]
        k = 1
        while (k < self.ell-1 or len(set(num_comps)) > 1) and k < self.N:
            G = self.Gknn_unweighted(k)
            NC_k = nx.number_connected_components(G)
            num_comps = [num_comps[i+1] if i < self.ell-1 else NC_k for i in range(self.ell)]
            k += 1
        return max(0, k-1)
    
    def Gknn_unweighted(self, k = None):
        if k is None: k = self.k
        A = kneighbors_graph(self.X, k, mode='distance', include_self = False).toarray()
        for i in range(self.N):
            for j in range(self.N):
                A[i, j] = max(A[i, j], A[j, i])
        G = nx.from_numpy_array(A)
        return G
    
    def Gknn_weighted(self, density = None, node_size = 25, width = 3):
        # points X on manifold of dimension n. Connect each point to its nearest n_nbrs, not including itself
        if self.Gknn is None:
            if density is None:
                f_est = f_estimator(self.n, self.X, kernel = self.kernel)
                density = f_est.density()
            alph = self.alpha()
            A = kneighbors_graph(self.X, self.k, mode='distance', include_self = False).toarray()
            for i in range(self.N):
                for j in range(self.N):
                    A[i, j] = max(A[i, j], A[j, i])
                    A[i, j] *= alph*(max(density[i], density[j]))**(1./self.n)
            self.Gknn = nx.from_numpy_array(A)
        return self.Gknn

    def geo_dist_pair(G_nodes):
        G = G_nodes[0]
        u = G_nodes[1]
        v = G_nodes[2]
        if nx.has_path(G, u, v):
            #return nx.shortest_path_length(G, u, v)
            return nx.dijkstra_path_length(G, u, v)
        else:
            return np.inf
    
    def distance_matrix(self, print_progress = True):
        if self.Gknn is None:
            self.Gknn = self.Gknn_weighted()
        G_nodes_list = [(self.Gknn, i, j) for i in range(self.N) for j in range(i)]
        with mp.Pool(mp.cpu_count()) as p:
            density_geodesic_dist = p.map(Geodist_estimator.geo_dist_pair, G_nodes_list)
        density_geodesic_dist = Geodist_estimator.reshape(density_geodesic_dist, self.N)
        return density_geodesic_dist
    
    def plot_Gknn_unweighted(self, node_size = 25):
        assert self.X.shape[1] == 2
        if self.Gknn is None:
            G = self.Gknn_unweighted(self.k)
        else:
            G = self.Gknn
        plt.rcParams['figure.figsize'] = (10, 10)
        plt.axis('equal')
        plt.axis('off')
        for u in G.nodes():
            G.nodes[u]['pos'] = self.X[u]
        nx.draw(G, pos = nx.get_node_attributes(G, 'pos'), node_size = node_size)
            
    def plot_Gknn_weighted(self, node_size  = 25, width = 3):
        assert self.X.shape[1] == 2
        if self.Gknn is None:
            self.Gknn = self.Gknn_weighted()
        for u in self.Gknn.nodes():
            self.Gknn.nodes[u]['pos'] = self.X[u]
        edges,weights = zip(*nx.get_edge_attributes(self.Gknn,'weight').items())
        
        fig, ax = plt.subplots()
        ax.axis('equal')
        ax.axis('off')
        
        cmap = plt.cm.rainbow
        norm = matplotlib.colors.Normalize()
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        nx.draw(self.Gknn, pos = nx.get_node_attributes(self.Gknn, 'pos'), edgelist = edges, width = width, edge_color= norm(weights), edge_cmap = cmap, node_size = node_size, node_color = 'k', ax = ax)
        fig.colorbar(sm, ax = ax)
        
    def plot_Gknn_components(self, kvals):
        num_components = [nx.number_connected_components(self.Gknn_unweighted(k)) for k in kvals]
        plt.plot(kvals, num_components)
        plt.hlines(y = num_components[-1], xmin = kvals[0], xmax = kvals[-1], color = 'r', linestyles='--')
        plt.xlabel(r'$k$')
        plt.ylabel(r'Number of components in $G_{kNN}(X)$')
        plt.show()
        return num_components
    
    def reshape(arr, N):
        # Reshape 1D arr into NxN array
        A = np.zeros((N, N))
        k = 0
        for i in range(N):
            for j in range(i):
                A[i, j] = arr[k]
                A[j, i] = A[i, j]
                k += 1
        return A


'''
Other Distance Matrices
'''
def alpha(n, N):
    '''
    Scaling factor as number of points N --> infty, in a manifold of dimension n.
    '''
    alpha = (N/(math.log(N)*(math.log(N) + (n-1)*math.log(math.log(N)))))**(1.0/n)
    return alpha
    
def density_weighted_distance(n, points = None, distance_matrix = None, density = None, kernel = f_estimator.biweight):
    assert points is not None or distance_matrix is not None, "Must enter either an array of points or distance_matrix"
    assert density is not None or (n is not None and kernel is not None)
    if points is not None:
        N = points.shape[0]
    else:
        N = distance_matrix.shape[0]
    if density is None:
        f_est = f_estimator(n, points, distance_matrix, kernel)
        density = f_est.density()
    if points is not None:
        modified_dist_matrix = np.zeros((N,N))
        for i, x in enumerate(points):
            for j, y in enumerate(points[:i]):
                modified_dist_matrix[i, j] = np.linalg.norm(x - y)*alpha(n, N)    # the missing factor 2 is to account for the fact that gudhi adds an edge btwn x and y if d(x,y) <= t, not 2t
                modified_dist_matrix[i, j] /= (density[i]**(-1.0/n) + density[j]**(-1.0/n))
                modified_dist_matrix[j, i] = modified_dist_matrix[i, j]
    if distance_matrix is not None:
        modified_dist_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(i):
                modified_dist_matrix[i, j] = distance_matrix[i, j]*alpha(n, N)
                modified_dist_matrix[i, j] /= (density[i]**(-1.0/n) + density[j]**(-1.0/n))
                modified_dist_matrix[j, i] = modified_dist_matrix[i, j]
    return modified_dist_matrix
    
def knn_distance(X):
    D = squareform(pdist(X))    # NxN euclidean distance matrix (np.array)
    N = X.shape[0]
    knnD = np.empty((N, N))
    for i in range(N):
        row = D[i]
        indices_sort = np.argsort(row)
        ranked_nbrs = np.zeros((N))
        for j, x in enumerate(indices_sort):
            ranked_nbrs[x] = j
        knnD[i, :] = ranked_nbrs
    return knnD
       
'''
Visualizations
'''

def plot_density(points, n = None, kernel = f_estimator.biweight, density = None, s = 3):
    # points is a Nxm array, where N is the number of points and m is the ambient dimension
    assert density is not None or (n is not None and kernel is not None)
    if density is None:
        f_est = f_estimator(n, points, kernel = kernel)
        density = f_est.density()
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize()
    color = cmap(norm(density))
    m = np.shape(points)[1]
    assert m == 3 or m == 2
    fig, ax = plt.subplots()
    ax.axis('equal')
    ax.axis('off')
    if m == 3:
        ax = plt.axes(projection = '3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s = s, color = color)
    else:
        xlowerlim = min(points[:, 0])
        xupperlim = max(points[:, 0])
        ylowerlim = min(points[:, 1])
        yupperlim = max(points[:, 1])
        eps = .2
        ax.set_xlim([xlowerlim*(1 + eps), xupperlim*(1 + eps)])
        ax.set_ylim([ylowerlim*(1 + eps), yupperlim*(1 + eps)])
        ax.scatter(points[:, 0], points[:, 1], s = s, color = color)
        
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(sm)
    plt.show()

def plot_1skeleton(points, distance_matrix, t):
    m = points.shape[1] # ambient dimension
    plt.axis('equal')
    plt.axis('off')
    assert m == 3 or m == 2
    if  m == 2:
        G = nx.Graph()
        for i, x in enumerate(points):
            G.add_node(i, pos = x)
            for j in range(i):
                if distance_matrix[i, j] < t:
                    G.add_edge(i, j)
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size = 10, node_color = 'k')
    else:
        fig, ax = plt.subplots()
        ax = plt.axes(projection = '3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s = 3)
        for i, x in enumerate(points):
            for j in range(i):
                y = points[j]
                if distance_matrix[i, j] < t:
                    ax.plot([x[0], y[0]], [x[1], y[1]], [x[2], y[2]], c = 'black')
        
#################################################################################
'''
Filtrations
'''

def DVR(n, k = None, points = None, distance_matrix = None, max_dimension = None, density = None, kernel = f_estimator.biweight, ell = 6, print_progress = True):
    assert points is not None or distance_matrix is not None, "Must enter an array of points or distance_matrix"
    geo_est = Geodist_estimator(points, n, k, ell, kernel)
    new_distance = geo_est.distance_matrix(print_progress)
    DVR_cpx = gd.RipsComplex(distance_matrix = new_distance).create_simplex_tree(max_dimension = max_dimension)
    return DVR_cpx

def density_weighted(n, points = None, max_dimension = None, distance_matrix = None, density = None, kernel = f_estimator.biweight):
    new_distance = density_weighted_distance(n, points, distance_matrix, density, kernel)
    density_weighted_cpx = gd.RipsComplex(distance_matrix = new_distance).create_simplex_tree(max_dimension = max_dimension)
    return density_weighted_cpx

def KNN(X, max_dimension = None):
    knnD = knn_distance(X)
    knn_cpx = gd.RipsComplex(distance_matrix = knnD).create_simplex_tree(max_dimension = max_dimension)
    return knn_cpx
