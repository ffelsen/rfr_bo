import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
from scipy import stats 
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances, r2_score, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering as AC
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from copy import copy
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading
from tqdm import tqdm
import warnings
from paretoset import paretoset
warnings.filterwarnings("ignore")

from sklearn.gaussian_process import GaussianProcessRegressor
from treeple import ObliqueRandomForestRegressor, ObliqueRandomForestClassifier, RandomForestRegressor, RandomForestClassifier, PatchObliqueRandomForestRegressor,  PatchObliqueRandomForestClassifier


from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


def get_N_modes(data):
    kde = gaussian_kde(data, bw_method=gaussian_kde(data).scotts_factor()*0.6)
    lims = [np.min(data), np.max(data)]
    kde_vals = kde(np.linspace(*lims,200))
    peaks = find_peaks(kde_vals)[0]
    return len(peaks)

def get_min_dist(xt,grid):
    ds = pairwise_distances(np.vstack([xt.reshape(1,-1),grid]))
    ds = ds[0,1:]
    return np.min(ds)


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N 
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points)) # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int') # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf') # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected 
    points_left = np.delete(points_left, selected) # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i-1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left])**2).sum(-1) # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point, 
                                        dists[points_left]) # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]

class BO_RFR:
    def __init__(self, X, y, grid, f, x_test, acq = 'BO', est_labels = 'known', class_method = 'ORFC', reg_method = 'ORFC', gpr = False, gl = None,
                 save_history = True, N_batch = 1, cl_weight = 0.3, k_init = 2, k_adaptive = True):
        self.X = X
        self.y = y
        self.N_batch = N_batch
        if self.N_batch > 1:
            self.batch = True
        else:
            self.batch = False
        self.grid = grid
        self.f = f
        self.reg_models = []
        self.cl_models = []
        self.save_history = save_history
        if self.save_history:
            self.Xs = [X]
            self.ys = [y]
        self.acq = acq
        self.est_labels = est_labels
        self.class_method = class_method
        self.reg_method = reg_method
        self.gpr = gpr
        self.scores = []
        self.x_test = x_test
        self.gl = gl
        print('generating test data')
        self.ref = np.array([self.f(xi) for xi in tqdm(self.x_test)]).ravel()
        self.n_iters = 0
        self.cl_weight = cl_weight
        self.k_init = k_init
        self.k_adaptive = k_adaptive
    
    def get_labels(self):

        if self.k_adaptive and self.est_labels in ['GM','AC']:
            self.k_init = get_N_modes(self.y.ravel())
        
        if self.est_labels == 'GM':
            gm = BayesianGaussianMixture(n_components=self.k_init, n_init=100, init_params='random').fit(self.y.reshape(-1,1))
            labels = gm.predict(self.y.reshape(-1,1)).ravel()
        elif self.est_labels == 'AC':
            knn = kneighbors_graph(self.X, n_neighbors=int(np.floor(self.X.shape[0]/2.)), ).toarray()
            acl = AC(n_clusters=self.k_init, connectivity=knn, linkage='single').fit(self.y.reshape(-1, 1))
            labels = acl.labels_
        elif self.est_labels == 'known':
            labels = np.array([self.gl(xi) for xi in self.X])
        else:
            print('No valid label method provided!\nValid options: GM, AC, known')
        self.labels = labels
        
        return labels 
        
    def fit_model(self):
        if self.gpr:
            self.model = GaussianProcessRegressor().fit(self.X, self.y)
        else:
            if self.reg_method == 'RFC':
                self.model = RandomForestRegressor(n_jobs = -1,).fit(self.X, self.y)
            elif self.reg_method == 'ORFC':
                self.model = ObliqueRandomForestRegressor(n_jobs = -1,).fit(self.X, self.y)
            elif self.reg_method == 'PORFC':
                self.model = PatchObliqueRandomForestRegressor(n_jobs = -1,).fit(self.X, self.y)
            else:
                print('No valid regressio method selected! valid options are: RFC, ORFC, PORFC')
    
    def get_uncertainty(self, x):
        if self.gpr:
            _, stds = self.model.predict(x, return_std=True)
        else:
            preds = np.vstack([est.predict(x) for est in self.model.estimators_])
            stds = np.std(preds,axis=0)
            stds = MinMaxScaler().fit_transform(stds.reshape(-1,1)).ravel()
        return stds
    
    def get_uncertainty_class(self, x):
        prob_vars = self.classifier.predict_proba(x).var(1)
        prob_vars = MinMaxScaler().fit_transform(prob_vars.reshape(-1,1)).ravel()
        return 1-prob_vars

    def get_ds(self, x):
        
        nn = NearestNeighbors(n_neighbors=1).fit(self.X)
        ds = nn.kneighbors(x)[0].ravel()
        ds = MinMaxScaler().fit_transform(np.array(ds).reshape(-1,1)).ravel()
        return ds
    
    def get_acc(self): 
        if self.gpr:
            stds = self.get_uncertainty(self.grid)
            return stds # max uncertainty
        else:
            ds = self.get_ds(self.grid)**2
            stds_cl = self.get_uncertainty_class(self.grid)
            stds = self.get_uncertainty(self.grid)
    

        if self.batch:
            return stds*ds, stds_cl
        else:    
            return stds*ds + self.cl_weight * stds_cl
    
    def fit_classifier(self):
        if self.class_method == 'RFC':
            self.classifier = RandomForestClassifier(n_jobs = -1, n_estimators=100).fit(self.X, self.labels)
        elif self.class_method == 'ORFC':
            self.classifier = ObliqueRandomForestClassifier(n_jobs = -1, n_estimators=100).fit(self.X, self.labels)
        elif self.class_method == 'PORFC':
            self.classifier = PatchObliqueRandomForestClassifier(n_jobs = -1, n_estimators=100).fit(self.X, self.labels)
        elif self.class_method == 'SVC':
            self.classifier = SVC(kernel='rbf',gamma=1, probability=True, C=1000).fit(self.X, self.labels)
        elif self.class_method == 'LSC':
            if self.X.shape[0] <= 7:
                nn = 4
            else:
                nn = 7
            self.classifier = LabelSpreading(kernel='rbf', n_neighbors=nn, n_jobs = -1, gamma = 1, alpha = 0.2).fit(self.X, self.labels)
        else:
            print('No valid classifier provided!\nValid options: RFC, SVC, LSC')
            
    def get_next_point(self):
        if self.acq == 'SF' and self.batch == False:
            ds = self.get_ds(self.grid)
            next_point = self.grid[np.argmax(ds)]
        elif self.acq == 'random' and self.batch == False:
            index = np.random.randint(0, len(self.grid))
            next_point = self.grid[index]
        elif self.acq == 'BO' and self.batch == False:
            next_point = self.grid[np.argmax(self.get_acc())]
        elif self.acq == 'BO' and self.batch == True:
            acc = self.get_acc()
            dat = np.vstack([acci.ravel() for acci in acc]).T
            pf = paretoset(dat,['max','max'])
            if len(pf[pf==True]) < self.N_batch:
                inds = np.where(pf==True)[0] 
                n_missing = self.N_batch - len(pf[pf==True])
                inds2 = np.where(pf==False)[0]
                inds2 = np.random.choice(inds2, n_missing, replace=False)
                inds = np.hstack([inds,inds2])
        
            elif len(pf[pf==True]) == self.N_batch:
                inds = np.where(pf==True)[0]
            else:
                batch = fps(dat[pf], self.N_batch)
                inds = []
                for i in batch:
                    ds = np.linalg.norm(dat-i.reshape(1,-1),axis=1)
                    inds.append(np.argmin(ds))
            next_point = self.grid[inds]
        else:
            print('No valid acquisition method provided!\nValid options: SF, random, BO\nWarning: Batch sampling is currently only implemented for BO')
        return next_point

    def run(self, n_iter):
        print('running iterative exploration:')
        self.fit_model()
        self.get_labels()
        self.fit_classifier()
        self.scores.append(self.get_r2())        
        for i in tqdm(range(n_iter)):
            next_point = self.get_next_point()

            if self.batch:
                next_y = np.array([self.f(ni) for ni in next_point]).ravel()
                self.X = np.vstack([self.X, next_point])
                self.y = np.hstack([self.y, next_y])
            else:
                next_y = self.f(next_point.reshape(1,-1))
                self.X = np.vstack([self.X, next_point.reshape(1,-1)])
                self.y = np.append(self.y, next_y)

            self.fit_model()
            self.get_labels()
            self.fit_classifier()
            self.n_iters += 1
            if self.save_history:
                self.reg_models.append(self.model)
                self.cl_models.append(self.classifier)
                self.Xs.append(self.X)
                self.ys.append(self.y)
            self.scores.append(self.get_r2())

    def get_rmse(self):
        pred = self.model.predict(self.x_test).ravel()
        E = self.ref - pred
        RMSE = np.sqrt(np.mean(E**2))
        return RMSE

    def get_r2(self):
        pred = self.model.predict(self.x_test).ravel()
        
        return r2_score(pred, self.ref)

    def get_mape(self):
        pred = self.model.predict(self.x_test).ravel()
        return mean_absolute_percentage_error(pred, self.ref)
