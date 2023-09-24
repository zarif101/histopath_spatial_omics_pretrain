from torch.utils.data import DataLoader,Dataset
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
from patchesdataset import *
from data import *

def create_patches_dataset(
    training_fold,
    patch_size=512,
    binary=True,
    transform=None,
    gene_subset=None
):
    if type(training_fold) is str:
        dataset = load_training_data(training_fold)
    else:
        dataset: SpatialOmicsSlide = training_fold

    if dataset.gene_ex is not None:
        patch_xys, labels = preprocess_wsi(dataset, patch_size, binary,center_on_patches=True,resample_aggregation='False')

        if gene_subset is not None:
            subset_indexes = [dataset.genes.index(
                gene) for gene in gene_subset if gene in dataset.genes]
            labels = labels[:, subset_indexes]
            genes = [gene for gene in gene_subset if gene in dataset.genes]
        else:
            genes = dataset.genes

    else:
        raise NotImplementedError("Unlabeled slides are not accounted for yet")

    image_pt = torch.from_numpy(dataset.image).permute(2, 0, 1) / 255.0

    return PatchesDataset(image_pt, labels, patch_xys, patch_size, transform=None, genes=genes)

class MultiPatchDataset(Dataset):
    def __init__(self, slide_ids, patch_size):
        self.slide_ids=slide_ids
        self.patch_size=patch_size
        gene_subset=np.load('../../spatial_omics/code/out/filtered_gene_list.npy')#top 1000 most variable by FSV
        self.dsets=[]
        for slide_id in slide_ids:
            dset=create_patches_dataset(slide_id,binary=False,patch_size=patch_size,gene_subset=gene_subset)
            self.dsets.append(dset)
        total_length=0
        for d in self.dsets:
            total_length+=d.__len__()
        self.length=total_length
    def __len__(self,):
        return self.length
    def __getitem__(self,i):
        length=self.length
        dset=0
        while self.dsets[dset].__len__() - 1 < i:
            i = i - self.dsets[dset].__len__()
            dset+=1
        return self.dsets[dset].__getitem__(i)

