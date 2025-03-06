# ternary plot
# !pip install python-ternary
import ternary
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations
import pandas as pd
import matplotlib.patheffects as PathEffects
from sklearn.svm import SVC
import pandas as pd


def plot_mixture_3d(list_plot, labels, heatf = [], title="", lines_points=[],color=[], add_labels_points=[]):
    # Scatter Plot
    scale = 1
    figure, tax = ternary.figure(scale=50)
    figure.set_figheight(5.5)
    figure.set_figwidth(6.0)
    if heatf!=[]:
        tax.heatmapf(heatf, boundary=True, style="triangular")
        figure.set_figwidth(7.5)
    for i,ps in enumerate(list_plot):
        if color==[]: color_points=[f"C{i+1}"]*len(ps)
        else: color_points = color
        tax.scatter(np.array(ps)*50, marker='o', color=color_points, label = labels[i], alpha=1, s=50)
        for j,p in enumerate(ps):
            if len(add_labels_points)>0: tax.annotate(str(add_labels_points[j]), np.array(p)*50,fontsize=20)
    tax.set_axis_limits((1,1,1))
    tax.clear_matplotlib_ticks()
    tax.get_axes().axis('off')
    tax.gridlines(multiple=10, color="black")
    tax.legend()
    tax.set_title(title)
    if lines_points!=[]:
        for p1,p2 in lines_points:
            tax.line( [int(round(x)) for x in np.array(p1)*50], [int(round(x)) for x in np.array(p2)*50], 
                      linewidth=2., color='gray', linestyle="-")
    tax.show()


# Quarternary plot: https://stackoverflow.com/questions/57467943/how-to-make-3d-4-variable-ternary-pyramid-plot-in-r-or-python

def _plot_ax(ax):               #plot tetrahedral outline
    verts=[[0,0,0],
     [1,0,0],
     [0.5,np.sqrt(3)/2,0],
     [0.5,0.28867513, 0.81649658]]
    lines=combinations(verts,2)
    for x in lines:
        line=np.transpose(np.array(x))
        ax.plot3D(line[0],line[1],line[2],c='0')

def _get_cartesian_array_from_barycentric(b):      #tranform from "barycentric" composition space to cartesian coordinates
    verts=[[0,0,0],
         [1,0,0],
         [0.5,np.sqrt(3)/2,0],
         [0.5,0.28867513, 0.81649658]]

    #create transformation array vis https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    t = np.transpose(np.array(verts))        
    t_array=np.array([t.dot(x) for x in b]) #apply transform to all points

    return t_array

def _plot_3d_tern(ax,df,c='1',alpha=0.3,s=40,alphas_sing=[]): #use function "get_cartesian_array_from_barycentric" to plot the scatter points
#args are b=dataframe to plot and c=scatter point color
    bary_arr=df.values
    cartesian_points=_get_cartesian_array_from_barycentric(bary_arr)
    if alphas_sing!=[]:
        ax.scatter(cartesian_points[:,0],cartesian_points[:,1],cartesian_points[:,2],c=alphas_sing,s=s,ec='k', cmap = 'tab10', vmin=0, vmax=10)
    else:
        ax.scatter(cartesian_points[:,0],cartesian_points[:,1],cartesian_points[:,2],c=c,alpha=alpha,s=s,ec='k', cmap = 'tab10', vmin=0, vmax=10)

        
def _label_components(ax,labels):  #create labels of each vertices of the simplex
    a=(np.array([0.1,-0.05,0.1,0])) # Barycentric coordinates of vertices (A or c1)
    b=(np.array([0.1,1.05,0.1,0.1])) # Barycentric coordinates of vertices (B or c2)
    c=(np.array([0,0,1,0.05])) # Barycentric coordinates of vertices (C or c3)
    d=(np.array([0,0,0,1.05])) # Barycentric coordinates of vertices (D or c3)
    cartesian_points=_get_cartesian_array_from_barycentric([a,b,c,d])
    
    cartesian_center = _get_cartesian_array_from_barycentric([[0.25,0.25,0.25,0.25]])
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='none')
    
    for point,label in zip(cartesian_points,labels):
        vec = point-cartesian_center.ravel()
        point += 0.2*vec
        txt = ax.text(point[0],point[1],point[2], label, size=16,ha='center', va='center', )
        txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])

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
        
def plot_mixture_4d(dfs_points,colors=[],labels_components = ["X1","X2","X3","X4"], alphas=[0.3]*20,sizes=[40]*20, alphas_sing=[],figsize=(5,5)):
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig) #Create a 3D plot in most recent version of matplot
    ax = fig.add_subplot(111, projection='3d')
    _plot_ax(ax) #call function to draw tetrahedral outline
    _label_components(ax,labels_components)
    for i,df in enumerate(dfs_points):
        #if len(colors)-1==i:
        #    _plot_3d_tern(ax,df,c=colors[i])
        #else:
        _plot_3d_tern(ax,df,c=colors[i],alpha=alphas[i],s=sizes[i],alphas_sing=alphas_sing)
    return ax

def plot_4d_multi(samples,ys, colors, boundary = False, cs = [], save=False):
    
    ncls = len(np.unique(ys[0]))
    # Read and fuse datasets
    df_all_features = pd.read_csv("raman/all_features_matrix2_3v1.csv")
    df_all_features = df_all_features[df_all_features.Fe>0]
    df_all_features = df_all_features[df_all_features.Pt>0]

    df_all_labels = pd.read_csv("raman/labels_matrix2_3_v1.csv")
    df_all_labels.index=df_all_labels.uuid
    df_all_features.index=df_all_features.uuid
    df_all_labels = df_all_labels.loc[df_all_features.uuid]

    cols=["Pt","Fe","Co","Ga","Mn","K","Ca"]
    cols_reduced=["Pt+Fe+Co+Ga","Mn","K","Ca"]
    cols_renamed = ["Pt+P1+P2+P3","P4","P5","P6"]
    df_all_features["Pt+Fe+Co+Ga"]=df_all_features["Pt"]+df_all_features["Fe"]+df_all_features["Co"]+df_all_features["Ga"]
    # Load candidates 
    candidates = np.load("raman/points_large.npy")
    df_candidates = pd.DataFrame(candidates,columns=cols)
    def dist_hp(x,w,b):
        x = x.ravel()                                                                                                                                                                                                
        return (np.dot(w.ravel(),x)+b)/np.linalg.norm(w)

    #cand_ds = np.array([dist_hp(candi, svc.coef_, svc.intercept_)[0] for candi in candidates])


    #labels_pred = svc.predict(df_all_features[cols]/100)

    #if boundary:
        #labels = labels_pred
    #else:
        #labels = np.array(labels)
    
    ax=plot_mixture_4d([df_all_features[cols_reduced]/100],
                    alphas=[0,],
                    sizes = [60 for i in range(ncls)],
                    colors=cs,
                    labels_components=cols_reduced,figsize=(6,6))
    for iy,y in enumerate(ys):
        # Fit svm
        svc=SVC(C=1000,kernel="linear",random_state=1,probability=True)
        label_df = pd.DataFrame({'spentID':samples,'label':y})


        labels = []
        feat_samps = df_all_features['spentID'].values
        for si, samp_feat in enumerate(feat_samps):
            for sj, sampj in enumerate(label_df['spentID'].values):
                if samp_feat == sampj:
                    labels.append(label_df['label'].values[sj])
        df_all_features["Pt+Fe+Co+Ga"]=df_all_features["Pt"]+df_all_features["Fe"]+df_all_features["Co"]+df_all_features["Ga"]
        cols=["Pt","Fe","Co","Ga","Mn","K","Ca"]
        cols_reduced=["Pt+Fe+Co+Ga","Mn","K","Ca"]
        cols_renamed = ["Pt+P1+P2+P3","P4","P5","P6"]

        svc.fit((df_all_features[cols]/100).values,labels)
        cand_ds = np.array([dist_hp(candi, svc.coef_,svc.intercept_)[0] for candi in candidates])

        boundary = candidates[abs(cand_ds)<5e-3]
        df_border = pd.DataFrame(boundary,columns=cols)
        df_border["Pt+Fe+Co+Ga"] = df_border["Pt"] + df_border["Fe"] + df_border["Co"] + df_border["Ga"]
        a = _get_cartesian_array_from_barycentric(df_border[cols_reduced].values)
        boundary_sparse = fps(a,3)
        ax.plot_trisurf(*boundary_sparse.T, alpha = 0.5, fc = colors[iy], ec = 'k', lw = 3)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.axis('off')

    eps = 0.03
    ticks = np.linspace(0.25,0.75,3)
    for ti in ticks:
        p = [ti,1-ti,0,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[0,-eps,0]])
        pc3 = pc1 + np.array([[0,-eps*1.3,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center',ha='right', fontsize = 8,zorder=-1)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [0,ti,1-ti,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[eps,0,0]])
        pc3 = pc1 + np.array([[1.3*eps,0,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center', fontsize = 8)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [0,0,ti,1-ti]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[0,0,eps]])
        pc3 = pc1 + np.array([[0,0,eps*1.3]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center', fontsize = 8,)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [ti,0,1-ti,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[-eps,0,0]])
        pc2 = pc1 + np.array([[-eps*1.3,0,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc2.ravel(),str(ti), va='center', fontsize = 8, zorder=-1)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    ax.set_proj_type('ortho')
    ax.view_init(azim=32,elev=21)
    if isinstance(save,str):
        plt.savefig('%s.png'%save,dpi=300)
    plt.show()
    return svc



def plot_4d(samples,y, boundary = False, cs = [], save=False):
    label_df = pd.DataFrame({'spentID':samples,'label':y})
    label_df.to_csv('label_df.csv')
    ncls = len(np.unique(y))
    # Read and fuse datasets
    df_all_features = pd.read_csv("raman/all_features_matrix2_3v1.csv")
    df_all_features = df_all_features[df_all_features.Fe>0]
    df_all_features = df_all_features[df_all_features.Pt>0]

    df_all_labels = pd.read_csv("raman/labels_matrix2_3_v1.csv")
    df_all_labels.index=df_all_labels.uuid
    df_all_features.index=df_all_features.uuid
    df_all_labels = df_all_labels.loc[df_all_features.uuid]

    # Fit svm
    svc=SVC(C=1000,kernel="linear",random_state=1,probability=True)

    label_df = pd.read_csv('label_df.csv')

    labels = []
    feat_samps = df_all_features['spentID'].values
    for si, samp_feat in enumerate(feat_samps):
        for sj, sampj in enumerate(label_df['spentID'].values):
            if samp_feat == sampj:
                labels.append(label_df['label'].values[sj])
    df_all_features["Pt+Fe+Co+Ga"]=df_all_features["Pt"]+df_all_features["Fe"]+df_all_features["Co"]+df_all_features["Ga"]
    cols=["Pt","Fe","Co","Ga","Mn","K","Ca"]
    cols_reduced=["Pt+Fe+Co+Ga","Mn","K","Ca"]
    cols_renamed = ["Pt+P1+P2+P3","P4","P5","P6"]
    
    svc.fit((df_all_features[cols]/100).values,labels)

    # Load candidates 
    candidates = np.load("raman/points_large.npy")
    df_candidates = pd.DataFrame(candidates,columns=cols)
    def dist_hp(x,w,b):
        x = x.ravel()                                                                                                                                                                                                
        return (np.dot(w.ravel(),x)+b)/np.linalg.norm(w)

    #cand_ds = np.array([dist_hp(candi, svc.coef_, svc.intercept_)[0] for candi in candidates])


    labels_pred = svc.predict(df_all_features[cols]/100)

    if boundary:
        labels = labels_pred
    else:
        labels = np.array(labels)
    
    ax=plot_mixture_4d([df_all_features[labels==i][cols_reduced]/100 for i in range(ncls)],
                    alphas=[1 for i in range(ncls)],
                    sizes = [60 for i in range(ncls)],
                    colors=cs,
                    labels_components=cols_reduced,figsize=(6,6))
    if boundary:
        cand_ds = np.array([dist_hp(candi, svc.coef_,svc.intercept_)[0] for candi in candidates])

        boundary = candidates[abs(cand_ds)<5e-3]
        df_border = pd.DataFrame(boundary,columns=cols)
        df_border["Pt+Fe+Co+Ga"] = df_border["Pt"] + df_border["Fe"] + df_border["Co"] + df_border["Ga"]
        a = _get_cartesian_array_from_barycentric(df_border[cols_reduced].values)
        boundary_sparse = fps(a,3)
        ax.plot_trisurf(*boundary_sparse.T, alpha = 0.5, fc = 'gray', ec = 'k', lw = 3)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.axis('off')

    eps = 0.03
    ticks = np.linspace(0.25,0.75,3)
    for ti in ticks:
        p = [ti,1-ti,0,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[0,-eps,0]])
        pc3 = pc1 + np.array([[0,-eps*1.3,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center',ha='right', fontsize = 8,zorder=-1)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [0,ti,1-ti,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[eps,0,0]])
        pc3 = pc1 + np.array([[1.3*eps,0,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center', fontsize = 8)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [0,0,ti,1-ti]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[0,0,eps]])
        pc3 = pc1 + np.array([[0,0,eps*1.3]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc3.ravel(),str(ti), va='center', fontsize = 8,)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    for ti in ticks:
        p = [ti,0,1-ti,0]
        pc1 = _get_cartesian_array_from_barycentric(np.array(p).reshape(1,-1))
        pc2 = pc1 + np.array([[-eps,0,0]])
        pc2 = pc1 + np.array([[-eps*1.3,0,0]])
        pc = np.vstack([pc1,pc2])
        ax.plot(*pc.T, c = 'k')
        txt = ax.text(*pc2.ravel(),str(ti), va='center', fontsize = 8, zorder=-1)
        txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    ax.set_proj_type('ortho')
    ax.view_init(azim=32,elev=21)
    if isinstance(save,str):
        plt.savefig('%s.png'%save,dpi=300)
    plt.show()
    return svc
