import os
import pickle
from models import *
import argparse
import torch
from torch_geometric.data import Data,Dataset,DataLoader
import random
from torch_geometric.data import Data
import pickle
import torch
from models import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import copy
from patchesdataset import *
from data import *
import itertools
import torch.nn.functional as F
import torch.nn as nn
import argparse
from wsi_transforms import wsi_training_transforms, wsi_testing_transforms
from torch_cluster.knn import knn_graph
import torch_geometric
import pyreadr
import torch.nn as nn

class ContrastiveModel(torch.nn.Module):
    def __init__(self, n_hidden, wsienc, ge_dim):
        super(ContrastiveModel, self).__init__()
        self.wsienc = wsienc
        #del self.wsienc.fc
        self.wsienc.fc = nn.Identity()
        self.wsi_project = nn.Linear(512, n_hidden)
        self.wsi_fc2 = nn.Linear(n_hidden, n_hidden)
        #self.wsi_fc3 = nn.Linear(n_hidden, n_hidden)
        
        self.ge_fc1 = nn.Linear(ge_dim, n_hidden)
        self.ge_fc2 = nn.Linear(n_hidden,n_hidden)
        self.ge_fc3 = nn.Linear(n_hidden,n_hidden)
        
        self.project = nn.Linear(n_hidden, n_hidden)
        
        self.dropout=nn.Dropout(p=0.4)
        
    def forward(self, wsi, ge):
        wsi = self.wsienc(wsi)
        wsi = self.dropout(wsi)
        wsi = F.relu(self.wsi_project(wsi))
        wsi = self.dropout(wsi)
        wsi = F.relu(self.wsi_fc2(wsi))
        wsi = self.dropout(wsi)
        #wsi = F.relu(self.wsi_fc3(wsi))
        wsi = F.relu(self.project(wsi))

        wsi = self.dropout(wsi)
        
        ge = F.relu(self.ge_fc1(ge))
        ge = F.relu(self.ge_fc2(ge))
        ge = self.dropout(ge)
        ge = F.relu(self.ge_fc3(ge))
        ge = F.relu(self.project(ge))
        ge = self.dropout(ge)
        return wsi, ge
    
def get_edges(xys):
    G=knn_graph(xys,k=16)#trying smaller number of neighbors
    #G=radius_graph(xy, r=2*np.sqrt(2), batch=None, loop=True)#trying radius graph
    G=torch_geometric.utils.add_remaining_self_loops(G)[0]
    xys=xys.detach().cpu()
    edges=G.detach().cpu().numpy().astype(int)
    return edges

def embed(wsi_directory):
    files = os.listdir(wsi_directory)
    save_dir='embeddings/tcga/autostained/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    patch_size = 224
    hidden_size=512
    items = [item.split('.')[0]+'.'+item.split('.')[1] for item in files]
    model_path = 'training_runs/2023_02_09_01_09_05_398415/model_ckhpt_epoch_117'
    model = torch.load(model_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    #IMPORTANT
    model.train(False)
    model.eval()
    
    #clinpath = '../../dnam_staging_model/COAD_pheno.rds'
    clin = pd.read_csv('clinical_coad.csv')
    
    #IMPORTANT
    for item in items:
        patches=np.load(wsi_directory+'/'+item+'.npy')
        info=pickle.load(open(wsi_directory+'/'+item+'.pkl','rb'))
        xys=torch.from_numpy(np.array(info[['x','y']])).to(device).float()
        try: 
            edges=get_edges(xys)
        except:
            continue# had an error on one image for some reason
        edges=torch.from_numpy(np.squeeze(edges))
        patch_embeddings=[]
        for i in range(0,len(patches)):
            patch = torch.from_numpy(np.squeeze(patches[i]))
            patch = np.transpose(patch,(2,0,1))
            patch = patch.unsqueeze(0).to(device)
            patch = patch/255 #IMPORTANT -- TRAINED WITH 0-1 RANGE IMAGES
            #patch = torch.from_numpy(patches[i]).to(device).float().reshape((-1,3,patch_size,patch_size))
            ge_dummy = torch.ones((1,1000)).to(device).float()
            with torch.no_grad():
                patch_embedding, __ = model(patch, ge_dummy)
                patch_embeddings.append(patch_embedding.detach().cpu().numpy().reshape(-1,hidden_size))
        patch_embeddings = np.array(patch_embeddings)
        patch_embeddings = torch.from_numpy(patch_embeddings)
        print(patch_embeddings.shape)
        embeddings=patch_embeddings
        #embeddings=torch.stack(patch_embeddings)
        #label=int(item.split('_')[2])
        tcga_id = item.split('-')
        tcga_id = tcga_id[0]+'-'+tcga_id[1]+'-'+tcga_id[2]
        print(tcga_id)
        met = clin[clin['submitter_id']==tcga_id]['ajcc_pathologic_n'].iloc[0]
        print(met)
        if '0' in met:
            label = 0
        else:
            label = 1
        data = Data(x=embeddings, edge_index=edges, 
                    edge_attr=None, y=label,batch=None)
        with open(save_dir+item+'.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=='__main__':
    embed('/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/multimodal/v2/WSI/COAD/patches_224/patches_224')
