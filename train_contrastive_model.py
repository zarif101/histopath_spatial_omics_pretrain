import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from patchesdataset import *
from data import *
from utils import *
#from models import ContrastiveModel
import torchvision
from torchvision import transforms
from losses import *
import os
import datetime
from autostain_utils import PatchDataset
from torch.utils.data import ConcatDataset

#need to load huge images
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

import torch.nn as nn
import torch.nn.functional as F
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
    
def train(epochs, model, train_loader, val_loader, device, transforms, n_hidden, optim,log):
    loss_fn = SupConLoss()
    wsi_ge_lambda = 0.5
    wsi_lambda = 1.5
    ge_lambda = 0.2
    train_losses = []
    val_losses = []
    if log:
        file=open(log+'/log.txt','a')
        file.write('Epochs, Train_Loss, Val_Loss, Val_WSI_Loss, Val_GE_Loss, Val_Crossmodal_Loss\n')
        file.close()
    for epoch in range(epochs):
        epoch_loss = 0
        wsi_epoch_loss=0
        ge_epoch_loss=0
        crossmodal_epoch_loss=0
        for idx, data in enumerate(train_loader):
            optim.zero_grad()
            wsi, ge = data
            wsi = wsi.to(device)
            ge = ge.float()
            ge[ge!=0] = torch.log2(ge[ge!=0]).float()#log2 transform
            ge = torch.nan_to_num(ge)
            ge = ge.to(device)
            ge_corrupt_mask = torch.cuda.FloatTensor(1000).uniform_() > 0.3
            ge_corrupt_mask_2 = torch.cuda.FloatTensor(1000).uniform_() > 0.3
            ge_corrupted = ge * ge_corrupt_mask # random mask GE corruption
            ge = ge * ge_corrupt_mask_2
            wsi = transforms(wsi)
            
            #for both wsi and ge, both views will now have "corruption" applied
            wsi_corrupted = transforms(wsi)
            wsi_embedding, ge_embedding = model(wsi, ge)
            wsi_embedding = torch.nn.functional.normalize(wsi_embedding,p=2.0)#l2 norm embeddings
            ge_embedding = torch.nn.functional.normalize(ge_embedding,p=2.0)
            
            wsi_embedding_corrupted, ge_embedding_corrupted = model(wsi_corrupted, ge_corrupted)
            wsi_embedding_corrupted = torch.nn.functional.normalize(wsi_embedding_corrupted,p=2.0)
            ge_embedding_corrupted = torch.nn.functional.normalize(ge_embedding_corrupted,p=2.0)
            
            wsi_embedding = wsi_embedding.reshape((-1, 1, n_hidden))
            wsi_embedding_corrupted = wsi_embedding_corrupted.reshape((-1, 1, n_hidden))
            ge_embedding = ge_embedding.reshape((-1,1,n_hidden))
            ge_embedding_corrupted = ge_embedding_corrupted.reshape((-1,1,n_hidden))
            
            wsi_views = torch.cat((wsi_embedding,wsi_embedding_corrupted),dim=1)
            ge_views = torch.cat((ge_embedding,ge_embedding_corrupted),dim=1)
            
            wsi_ge_views = torch.cat((wsi_embedding,ge_embedding),dim=1)
           
            wsi_loss = loss_fn(wsi_views)*wsi_lambda
            wsi_epoch_loss+=wsi_loss.item()
            ge_loss = loss_fn(ge_views)*ge_lambda
            wsi_ge_loss = loss_fn(wsi_ge_views)*wsi_ge_lambda
            
            ge_epoch_loss+=ge_loss.item()
            crossmodal_epoch_loss+=wsi_ge_loss.item()
            #wsi_ge_loss = 0
            loss = wsi_loss + ge_loss + wsi_ge_loss; epoch_loss+=loss.item()
            loss.backward()
            optim.step()
        val_loss,val_wsi_loss, val_ge_loss, val_crossmodal_loss = val(model, val_loader, device, transforms)
        crossmodal_epoch_loss/=len(train_loader)
        ge_epoch_loss/=len(train_loader)
        wsi_epoch_loss/=len(train_loader)
        epoch_loss /= len(train_loader)
        print('Epoch '+str(epoch)+' - Train Loss: '+str(epoch_loss)+' Val Loss: '+str(val_loss)+' WSI Val Loss: '+str(val_wsi_loss)+' GE Val Loss: '+str(val_ge_loss)+' Cross Val Loss: '+str(val_crossmodal_loss)+str('\n'))
        if log:
            file=open(log+'/log.txt','a')
            write_str = str(epoch)+', '+str(epoch_loss)+', '+str(val_loss)+', '+str(val_wsi_loss)+', '+str(val_ge_loss)+', '+str(val_crossmodal_loss)+'\n'
            file.write(write_str)
            file.close()
            torch.save(model, log+'/model_ckhpt_epoch_'+str(epoch))
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        for param in model.parameters():
            if param.requires_grad == False:
                print('HERERER',epoch)
        if epoch > 40: 
            for param in model.wsienc.parameters():
                param.requires_grad = True
            #unfreeze resnet
    return train_losses, val_losses
            
def val(model, val_loader, device, transforms):
    loss_fn = SupConLoss()
    wsi_ge_lambda = 0.5
    wsi_lambda = 1.5
    ge_lambda = 0.2
    model.eval()
    epoch_loss = 0 
    wsi_total_loss=0
    crossmodal_total_loss=0
    ge_total_loss=0
    with torch.no_grad():
        for idx, data in enumerate(val_loader):
            wsi, ge = data
            wsi = wsi.to(device)
            ge = ge.float()
            ge[ge!=0] = torch.log2(ge[ge!=0]).float()
            ge = torch.nan_to_num(ge)
            ge = ge.to(device)

            ge_corrupt_mask = torch.cuda.FloatTensor(1000).uniform_() > 0.3
            ge_corrupt_mask_2 = torch.cuda.FloatTensor(1000).uniform_() > 0.3
            ge_corrupted = ge * ge_corrupt_mask # random mask GE corruption
            
            ge = ge * ge_corrupt_mask_2
            wsi = transforms(wsi)
            
            wsi_corrupted = transforms(wsi)
            wsi_embedding, ge_embedding = model(wsi, ge)
            wsi_embedding = torch.nn.functional.normalize(wsi_embedding,p=2.0)#l2 norm embeddings
            ge_embedding = torch.nn.functional.normalize(ge_embedding,p=2.0)
            
            wsi_embedding_corrupted, ge_embedding_corrupted = model(wsi_corrupted, ge_corrupted)
            wsi_embedding_corrupted = torch.nn.functional.normalize(wsi_embedding_corrupted,p=2.0)
            ge_embedding_corrupted = torch.nn.functional.normalize(ge_embedding_corrupted,p=2.0)
            
            wsi_embedding = wsi_embedding.reshape((-1, 1, n_hidden))
            wsi_embedding_corrupted = wsi_embedding_corrupted.reshape((-1, 1, n_hidden))
            ge_embedding = ge_embedding.reshape((-1,1,n_hidden))
            ge_embedding_corrupted = ge_embedding_corrupted.reshape((-1,1,n_hidden))

            wsi_views = torch.cat((wsi_embedding,wsi_embedding_corrupted),dim=1)
            ge_views = torch.cat((ge_embedding,ge_embedding_corrupted),dim=1)

            wsi_ge_views = torch.cat((wsi_embedding,ge_embedding),dim=1)
            wsi_loss = loss_fn(wsi_views)*wsi_lambda
            ge_loss = loss_fn(ge_views)*ge_lambda
            wsi_ge_loss = loss_fn(wsi_ge_views)*wsi_ge_lambda
            #wsi_ge_loss = 0
            wsi_total_loss+=wsi_loss.item()
            ge_total_loss+=ge_loss.item()
            crossmodal_total_loss+=wsi_ge_loss.item()
            loss = wsi_loss + ge_loss + wsi_ge_loss; epoch_loss+=loss.item()
    epoch_loss /= len(val_loader)
    wsi_total_loss/=len(val_loader)
    ge_total_loss/=len(val_loader)
    crossmodal_total_loss/=len(val_loader)
    model.train()
    return epoch_loss,wsi_total_loss, ge_total_loss, crossmodal_total_loss

def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)
    return model

def train_model(epochs, batch_size, train_samples, val_samples, patch_size, n_hidden, lr,
               log='training_runs',freeze_resnet=True):
    
    device = torch.device('cuda')
    train_sets=[]
    val_sets = []
    gene_subset=list(np.load('../../spatial_omics/code/out/filtered_gene_list.npy'))
    for sample in train_samples:
        path = 'autostained_extra/new_seq/'+sample+'.pkl'
        slide = pickle.load(open(path,'rb'))
        slide.genes = gene_subset
        slide = slide.select_genes(gene_subset)
        ds = PatchDataset(slide,patch_size,1,None,device)
        train_sets.append(ds)
    for sample in val_samples:
        path = 'autostained_extra/new_seq/'+sample+'.pkl'
        slide = pickle.load(open(path,'rb'))
        slide.genes = gene_subset
        slide = slide.select_genes(gene_subset)
        ds = PatchDataset(slide,patch_size,1,None,device)
        val_sets.append(ds) 
    train_dataset = ConcatDataset(train_sets)
    val_dataset = ConcatDataset(val_sets)
    print(len(train_dataset))
    print(len(val_dataset))
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=True)
    
    wsienc = torchvision.models.resnet18(pretrained=False)
    wsienc.fc=torch.nn.Identity()
    weights=torch.load('extra_space/ciga_pretrained_resnet18.ckpt')
    state_dict = weights['state_dict']
    for key in list(state_dict.keys()):
        state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)
    
    wsienc=load_model_weights(wsienc,state_dict)
    if freeze_resnet == True:
        for param in wsienc.parameters():
            param.requires_grad = False #freezing resnet weights
    model = ContrastiveModel(n_hidden, wsienc, 1000).to(device)
    wsi_transforms_old = torch.nn.Sequential(
    transforms.RandomRotation(degrees=270),
    transforms.RandomGrayscale(p=0.25),
    transforms.RandomSolarize(threshold=0.4,p=0.3)
        )
    wsi_transforms = torch.nn.Sequential(
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.25),
        transforms.RandomSolarize(threshold=0.4,p=0.3),
        transforms.RandomRotation(degrees=270)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=1e-5)
    if log:
        now = datetime.datetime.now()
        now = str(now).replace(' ','_').replace('.','_').replace('-','_').replace(':','_')
        os.mkdir(log+'/'+now)
        log = log+'/'+now
        f=open(log+'/params.txt','a')
        f.write(str(epochs)+' epochs, '+str(batch_size)+' batch_size, '+str(n_hidden)+' hidden, '+str(lr)+' lr, '+str(patch_size)+' patch_size')
        f.close()
    train(epochs, model, train_loader, val_loader, device,
          wsi_transforms, n_hidden, optimizer, log)
    
if __name__=='__main__':
    device=torch.device('cuda')
    epochs = 150
    batch_size = 8
    #train_samples = ['A1','B1']
    #val_samples = ['C1','D1']
    train_samples = ['092842','091759']
    val_samples = ['092534','092146']
    patch_size = 224
    n_hidden = 100
    lr = 0.00001
    train_model(epochs, batch_size, train_samples, val_samples, patch_size, n_hidden, lr)
