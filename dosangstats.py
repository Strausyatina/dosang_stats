import subprocess
import os
import pandas as pd
import csv
import scipy as scp
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.spatial import distance


# 1. GET DATA
# obtain csv dictionary from a given text file with target folders listed

def get_folders_dict(dictfilename):
    '''obtain all the pathes to files''' 
    folders_dict = {}
    with open (dictfilename, 'r') as f_source:
        for line in f_source:
            folder  = line.strip()
            name = folder.split('/')[-1]
            folders_dict[name] = {}
            for item in os.listdir(folder):
                if item.endswith("nods.csv"): folders_dict[name]['nods'] = folder + '/' + item
                elif item.endswith("paths.csv"): folders_dict[name]['paths'] = folder + '/' + item
                elif item.endswith("points.csv"): folders_dict[name]['points'] = folder + '/' + item
    # filter empty:
    bad_folders = []
    for item in folders_dict:
        if not folders_dict[item]: bad_folders.append(item)
    for item in bad_folders:
        del folders_dict[item]
    
    return folders_dict
  
def get_csv_dict(folders_dict):
    '''obtain all csv-s'''
    csv_dict = {}
    for item in folders_dict:
        csv_dict[item] = {}
        for jtem in folders_dict[item]:
            csv_dict[item][jtem] = pd.read_csv(folders_dict[item][jtem], encoding='cp1251') #"utf-8"
            
    return csv_dict
  

  
def find_condition_observation(comment): 
    '''looking for keywords about condition of path observation'''
    if isinstance(comment, unicode):
      comment = comment.encode("utf-8")
    if pd.isnull(comment):
      return "MIDPATH"
    elif "START" in comment:
      return "START"
    elif "STOP" in comment:
      return "STOP"
    elif "MEET" in comment:
      return "MEET"
    else:
      return "MIDPATH"
def idvect_to_condition_observation(id_vect):
    id_cond = ["MIDPATH" for i in range(len(id_vect))]
    id_set = {id_vect[0]}
    id_cond[0] = "START"
    id_cond[len(id_vect)-1] = "STOP"
    for i in range(1,(len(id_vect)-1)):
        if id_vect[i]!=id_vect[i-1] and id_vect[i]==id_vect[i+1]:
            id_cond[i] = "START"
            id_set = {id_vect[i]}
        elif id_vect[i]!=id_vect[i+1] and id_vect[i]==id_vect[i-1]:
            id_cond[i] = "STOP"
    
    return id_cond
    
def complement_csv_points(IN, points_csv, path_csv=None):
    '''add IndividualNumber of lizard and path info columns to each corresponding point'''
    points_compl = points_csv.sort_values(by=['date', 'time1', 'time2']).copy()
    points_compl['IN'] = pd.Series([IN for i in range(points_compl.shape[0])], index=points_compl.index)
    points_compl['pathness'] = pd.Series(points_compl.apply(lambda x: find_condition_observation(x['comment']), axis=1), index=points_compl.index)
    
    ddd = pd.DataFrame(np.nan, index=points_compl.index, columns=['ID_paths', 'time1_path', 'time2_path', 'length_path'])
    if path_csv is None:
      points_compl = pd.concat([points_compl, ddd], axis=1)
      return points_compl 
    
    for idx, x in points_compl.iterrows():
        a = path_csv.loc[path_csv['date'] == x['date']].loc[path_csv['timestart'] <=x ['time1']].loc[path_csv['timeend'] >= x['time1']]
        if not a.empty:
            #print(a); print(x); print()
            ddd.loc[idx] = a.loc[a.index[0],['ID','timestart','timeend','length']].values
            # NOT A GOOD SOLUTION TO TAKE FIRST ROW IF IT's INTERSECTION
            #ddd.loc[idx] = a.loc[0,['ID','timestart','timeend','length']].values
    points_compl = pd.concat([points_compl, ddd], axis=1)
    
    return points_compl

def precomplement_csv_nods(IN, nods_csv, path_csv):
    '''add IndividualNumber of lizard and path info columns to each corresponding point'''
    nods_compl = nods_csv.copy()
    nods_compl['IN'] = pd.Series([IN for i in range(nods_compl.shape[0])], index=nods_compl.index)
    nods_compl['pathness'] = pd.Series(idvect_to_condition_observation(nods_compl['ID'].values), index=nods_compl.index)
    
    zero_cols = pd.DataFrame(0, index=nods_compl.index, columns=["beg_end", "type", "partner", "res", "time1"]).astype(int)
    nan_cols = pd.DataFrame(np.nan, index=nods_compl.index, columns=["time2", "subtype"])
    nods_compl = pd.concat([nods_compl, zero_cols, nan_cols], axis=1) 
    
    ddd = pd.DataFrame(np.nan, index=nods_compl.index, columns=['ID_paths', 'date', 'time1_path', 'time2_path', 'length_path'])
    for idx, x in nods_compl.iterrows():
        a = path_csv.loc[path_csv['ID']==x['ID']]
        ddd.loc[idx] = a.loc[0,['ID','date','timestart','timeend','length']].values
    nods_compl = pd.concat([nods_compl, ddd], axis=1)    
    nods_compl[['date', 'time1']] = nods_compl[['date', 'time1']].astype(int)
    
    nods_compl.loc[nods_compl['pathness']=="START", "time1"] = nods_compl.loc[nods_compl['pathness']=="START", "time1_path"].astype(int) 
    nods_compl.loc[nods_compl['pathness']=="STOP", "time1"] = nods_compl.loc[nods_compl['pathness']=="STOP", "time2_path"].astype(int) 
    
    return nods_compl

def merge_points_nods(points_tab, nods_tab):
    #
    # WRITE THE MERGE
    # CODE HERE!!!!!!
    #
    return
  
def time_to_minuts(t):
    return int(t//(10**7)*60 + (t//(10**5))%100)
def minuts_to_time(m):
    return int(m//60*(10**7) + m%60*(10**5))
def date_to_minuts(d1, d, t):
    return int((d-d1)*60*24 + t//(10**7)*60 + (t//(10**5))%100)
  
def fill_time_vect(V):
    X = [time_to_minuts(v) for v in V]
    filling_mode = False
    filling_start = 0
    for i in range(1,len(X)):
      if X[i]==0 and not filling_mode:
        filling_mode = True
        filling_start = i-1
      elif X[i]!=0 and filling_mode:
        n = i - filling_start
        st = X[filling_start]
        et = X[i]
        mid = np.divide(np.arange(1.0, n, 1), [n])
        mid = np.multiply(np.subtract(et, st), mid)
        X[(filling_start+1):i] = np.rint(np.add(st, mid)).tolist()
        filling_mode = False
    return [minuts_to_time(x) for x in X]
        
        
def fill_time_gaps(tab):
    filled_tab = tab.copy()
    filled_tab['time1'] = fill_time_vect(filled_tab['time1'].values)
    return filled_tab
  
def complement_whole_csv(IN, csv_hub):
    if 'paths' not in csv_hub.keys():
      return complement_csv_points(IN, csv_hub["points"])
    elif 'points' not in csv_hub.keys():
      return fill_time_gaps(precomplement_csv_nods(IN, csv_hub["nods"], csv_hub["paths"]))
    else:
      points_tab = complement_csv_points(IN, csv_hub["points"], csv_hub["paths"])
      nods_tab = precomplement_csv_nods(IN, csv_hub["nods"], csv_hub["paths"])
      return fill_time_gaps(merge_points_nods(points_tab, nods_tab))
  
  
  
  
  
  
  
# (!) Rewrite ConvexHull (1)counting, (2)plotting and (3)area calculation
def ConvexHull_sq(csv_dict_name):  
    '''calculate convex hull area'''
    for_concat = []
    if 'nods' in csv_dict_name:
        nods_coords = csv_dict_name['nods'].filter(items=['coord_X', 'coord_Y'])
        for_concat.append(nods_coords)
    if 'points' in csv_dict_name:
        points_coords = csv_dict_name['points'].filter(items=['coord_X', 'coord_Y'])
        for_concat.append(points_coords)
        
    if not for_concat: return 0
    res = pd.concat(for_concat, ignore_index = True)
    
    plt.plot(res.coord_X, res.coord_Y, 'o')
    if len(res) <= 2: return 0
    
    hull = ConvexHull(res)
    #print(len(hull.points))
    #print(hull.vertices)
    #for hv in hull.vertices: print(res.coord_X[hv], res.coord_Y[hv])
    plt.plot(res.coord_X[hull.vertices], res.coord_Y[hull.vertices], 'r--', lw=2)
    #plt.ylim([-50,50])
    #plt.xlim([-50,50])
    
    for simplex in hull.simplices:
        plt.plot(res.coord_X[simplex], res.coord_Y[simplex], 'k-')
    plt.show()
    
    return (len(res), hull.area)  
    