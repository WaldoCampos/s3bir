"""
compute mAP given a result file
"""
import glob, os
import re
import numpy as np
import  matplotlib.pyplot as plt
regex = '[0-9]+'
def get_map(file):
    mAP = 0
    i = 0
    with open(file) as f:        
        for line in f :
            result = line.strip().split(',')            
            ranking = np.array([int(elem == result[0]) for elem in result[1:]])
            # print(ranking)
            inds = np.arange(1, len(ranking)+1)
            recall = np.cumsum(ranking) * ranking
            valid_inds = recall != 0
            inds = inds[valid_inds]
            recall = recall[valid_inds]
            precision = recall / inds
            AP = np.mean(precision)
            mAP += AP
            i +=1
    return mAP/i
            
            
            
             
if __name__ == '__main__' :
    # #folder = '/home/jsaavedr/Research/data/vete-gnn/labels'
    # folder = '/home/jsaavedr/Research/data/vete-gnn/labels_based_on_text_search'
    # dmAP = {}        
    # for file in glob.glob(os.path.join(folder, "*_GC*.csv")):            
    #     result =  re.findall(regex, file)
    #     it = int(result[0])
    #     dmAP[it] = get_map(file)
    
    # keys = list(dmAP.keys())
    # keys = sorted(keys)
    # its = []
    # mAPs = []
    # for it in keys :
    #     print('{}  {}' .format(it, dmAP[it]))
    #     its.append(it)
    #     mAPs.append(dmAP[it])
     
    # plt.plot(its, mAPs)
    # plt.show()
    file_path = '/home/wcampos/search_output.txt'
    final_metric = get_map(file_path)
    print(f"\n'mAP del modelo: {final_metric}")
    