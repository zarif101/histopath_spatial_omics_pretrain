from typing import List, Optional, Union
import torch
from data import load_training_data, SpatialOmicsSlide
import os
#from functools import cached_property
from typing import List, Optional, Union
import numpy as np

import torch
import torchvision.transforms.functional as TF
#from device import device
if torch.cuda.is_available():
    device = torch.device('cuda')


def preprocess_wsi(data: SpatialOmicsSlide, patch_size: int, binary: bool, center_on_patches: bool, resample_aggregation: str):
    """
    Data processing. Split into patches of size 512x512.
    Then, for each patch, find whether the expression of
    a given gene is above or below the median expression
    of that gene across the whole slide.
    """

    if not center_on_patches:
        top_left = min(data.gene_ex_X), min(data.gene_ex_Y)
        data = data.resample_grid(data.create_grid(patch_size, top_left), patch_size, resample_aggregation)

    gene_ex = data.gene_ex.to(device)
    xy = torch.stack([data.gene_ex_X, data.gene_ex_Y], dim=1)

    # select only valid xys
    if center_on_patches:
        valid_indexes = \
              (xy[:, 0] >= (patch_size // 2)) \
            & (xy[:, 1] >= (patch_size // 2)) \
            & (xy[:, 0] < (data.image.shape[1] - patch_size // 2)) \
            & (xy[:, 1] < (data.image.shape[0] - patch_size // 2))
    else:
        valid_indexes = \
              (xy[:, 0] >= 0) \
            & (xy[:, 1] >= 0) \
            & (xy[:, 0] < (data.image.shape[1] - patch_size)) \
            & (xy[:, 1] < (data.image.shape[0] - patch_size))
    gene_ex = gene_ex[valid_indexes]
    xy = xy[valid_indexes]

    if binary:
        median: torch.Tensor = torch.median(gene_ex, dim=0)[0]
        gene_ex_binary = (gene_ex > median).float()
        return xy, gene_ex_binary
    else:
        return xy, gene_ex


class PatchesDataset(torch.utils.data.Dataset):
    """
    A PatchesDataset is a bunch of patches with positional information and corresponding genes.
    This dataset format can be used for several slides in aggregate when training a model that is position-agnostic,
    or it can also be used for individual slides to train models that require positional information.
    To create a dataset of slides that need to be separated from each other, use a SlidesDataset.
    """

    def __init__(
        self,
        image: torch.Tensor,
        labels: Optional[torch.Tensor],
        patch_xys: torch.Tensor,
        patch_size: int,
        center_on_patches: bool,
        transform=None):
        super().__init__()

        self.image = image
        self.patch_xys = patch_xys
        self.labels = labels
        self.transform = transform
        self.patch_size = patch_size
        if labels is not None:
            self.n_genes = labels.shape[1]
        else:
            self.n_genes = -1

        self.center_on_patches = center_on_patches

        self.return_locs = True

    #@cached_property
    def patches(self):
        patches = torch.zeros((self.patch_xys.shape[0], 3, self.patch_size, self.patch_size), dtype=torch.float, device=device, requires_grad=False)
        for i, (x, y) in enumerate(self.patch_xys):
            patches[i] = self._patch(x, y)
        return patches

    def _patch(self, x, y):
        if self.center_on_patches:
            # Using in-place operations here sucks
            x = x - self.patch_size // 2
            y = y - self.patch_size // 2

        return self.image[:, y:y + self.patch_size, x:x + self.patch_size]

    def __len__(self):
        return self.patch_xys.shape[0]

    def __getitem__(self, index: int):
        x, y = self.patch_xys[index]

        patch = self._patch(x, y).to(device)
        if patch.shape[1] != self.patch_size:
            raise ValueError(f"Patch at {x}, {y} is {patch.shape} but expected {self.patch_size}x{self.patch_size}")

        assert patch.shape == (3, self.patch_size, self.patch_size)
        if self.transform is not None:
            patch = self.transform(patch)

        if self.labels is not None:
            label = self.labels[index].float()
        else:
            label = None
        
        loc = self.patch_xys[index]

        if self.return_locs:
            return patch, label, loc
        else:
            return patch, label

class PatchesDataset(torch.utils.data.Dataset):
    """
    A PatchesDataset is a bunch of patches with positional information and corresponding genes.
    This dataset format can be used for several slides in aggregate when training a model that is position-agnostic,
    or it can also be used for individual slides to train models that require positional information.
    To create a dataset of slides that need to be separated from each other, use a SlidesDataset.
    """

    def __init__(
        self,
        image: torch.Tensor, # or np.ndarray[uint8]
        labels: Optional[torch.Tensor],
        patch_xys: torch.Tensor,
        patch_size: int,
        transform=None,
        genes: List[str] = None,
        normalizer=None):
        super().__init__()

        if genes is None:
            raise ValueError("genes must be provided")

        self.image = image
        self.patch_xys = patch_xys
        self.labels = labels
        self.transform = transform
        self.patch_size = patch_size
        self.genes = genes
        self.normalizer = normalizer
        if labels is not None:
            self.n_genes = labels.shape[1]
        else:
            self.n_genes = -1

        self.return_locs = False

    def set_patch_size(self, patch_size: int):
        # del self.patches
        self.patch_size = patch_size

    #@cached_property
    def patches(self):
        patches = torch.zeros((self.patch_xys.shape[0], 3, self.patch_size, self.patch_size), dtype=torch.float, device=device, requires_grad=False)
        for i, (x, y) in enumerate(self.patch_xys):
            patches[i] = self._patch(x, y)
        return patches

    def _patch(self, x, y):
        # Using in-place operations here sucks
        x = x - self.patch_size // 2
        y = y - self.patch_size // 2

        if self.normalizer is not None:
            image = self.image[y:y + self.patch_size, x:x + self.patch_size, :]
            assert type(image) == np.ndarray
            image = self.normalizer.transform(image)
            image = TF.to_tensor(image)
        else:
            image = self.image[:, y:y + self.patch_size, x:x + self.patch_size]
        return image

    def __len__(self):
        return self.patch_xys.shape[0]

    def __getitem__(self, index: int):
        x, y = self.patch_xys[index]

        patch = self._patch(x, y).to(device)
        if patch.shape[1] != self.patch_size:
            raise ValueError(f"Patch at {x}, {y} is {patch.shape} but expected {self.patch_size}x{self.patch_size}")

        assert patch.shape == (3, self.patch_size, self.patch_size)
        if self.transform is not None:
            patch = self.transform(patch)

        if self.labels is not None:
            label = self.labels[index].float()
        else:
            label = None
        
        loc = self.patch_xys[index]

        if self.return_locs:
            return patch, label, loc
        else:
            return patch, label
class PatchesDatasetRandomVerticalFlip:
    """
    Performs a random vertical flip of a patches dataset. This does not
    change the order of the internal list of patches; it just changes the Y values
    relative to each other.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: PatchesDataset):
        if torch.rand(()) >= self.p:
            return sample

        max_y = sample.patch_xys[:, 1].max()
        new_xys = sample.patch_xys.clone()
        new_xys[:, 1] = max_y + sample.patch_size - new_xys[:, 1]
        # (B, C, *H*, W)
        new_patches = torch.flip(sample.patches, dims=(2,))
        return PatchesDataset(
            new_patches,
            sample.labels,
            new_xys,
            sample.patch_size,
            sample.transform
        )


class PatchesDatasetRandomHorizontalFlip:
    """
    Performs a random horizontal flip of a patches dataset. This does not
    change the order of the internal list of patches; it just changes the X values
    relative to each other.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample: PatchesDataset):
        if torch.rand(()) >= self.p:
            return sample

        max_x = sample.patch_xys[:, 0].max()
        new_xys = sample.patch_xys.clone()
        new_xys[:, 0] = max_x + sample.patch_size - new_xys[:, 0]
        # (B, C, H, *W*)
        new_patches = torch.flip(sample.patches, dims=(3,))
        return PatchesDataset(
            new_patches,
            sample.labels,
            new_xys,
            sample.patch_size,
            sample.transform,
        )


def create_patches_dataset_old(
    training_fold: Union[str, SpatialOmicsSlide],
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



class SlidesDataset(torch.utils.data.Dataset):
    """
    A SlidesDataset is a wrapper for a list of PatchesDatasets that are separated based on the slide they are associated with.
    Note: All slides must have the same number of genes.
    """

    def __init__(self, slides: List[PatchesDataset], transform=None):
        super().__init__()

        self.slides = slides
        self.n_genes = slides[0].n_genes
        self.transform = transform

        for slide in slides:
            assert slide.n_genes == self.n_genes, "All slides must have the same number of genes"

    def __len__(self):
        return len(self.slides)

    def __getitem__(self, index: int):
        slide = self.slides[index]
        if self.transform is not None:
            slide = self.transform(slide)

        return slide


def create_slides_dataset(training_folds, patch_size=512, binary=True, slide_transform=None, patch_transform=None, gene_subset=None):
    """
    Creates a SlidesDataset based on several SpatialOmicsSlide objects or strings representing the ID of the
    SpatialOmicsSlide to load. An optional transform may be provided, which is passed to each PatchesDataset.
    """

    return SlidesDataset(
        [create_patches_dataset(training_folds[i:i + 1], patch_size, binary,
                                patch_transform, gene_subset) for i in range(len(training_folds))],
        slide_transform,
    )
