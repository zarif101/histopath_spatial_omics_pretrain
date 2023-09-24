from dataclasses import dataclass
#from functools import cached_property
from typing import overload

import matplotlib
import numpy as np
import pandas as pd
import PIL.Image
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import cv2

@dataclass
class SpotLocations:
    image_x: torch.Tensor
    image_y: torch.Tensor
    row: torch.Tensor
    col: torch.Tensor
    dia: torch.Tensor

    def __mul__(self, scale_factor: float):
        if self.dia is None:
            dia = None
        else:
            dia = self.dia * scale_factor
        return SpotLocations(
            self.image_x * scale_factor,
            self.image_y * scale_factor,
            self.row,
            self.col,
            dia,
        )
    
    def __div__(self, dividend: float):
        if self.dia is None:
            dia = None
        else:
            dia = self.dia / dividend
        return SpotLocations(
            self.image_x / dividend,
            self.image_y / dividend,
            self.row,
            self.col,
            dia,
        )
    
    def select_subset(self, mask):
        return SpotLocations(
            self.image_x[mask],
            self.image_y[mask],
            self.row[mask],
            self.col[mask],
            self.dia[mask] if self.dia is not None else None,
        )

    def __len__(self):
        return self.image_x.shape[0]

_image_cache = {}

class Slide:
    def __init__(self, image_path: str, spot_locations: SpotLocations, spot_counts: torch.Tensor, genes: list):
        self.image_path = image_path
        self.spot_locations = spot_locations
        self.spot_counts = spot_counts
        self.genes = genes

    # split into quadrants (and their complements)
    def create_quadrants(self):
        spot_locations = self.spot_locations
        row = spot_locations.row
        col = spot_locations.col
        max_row = row.max()
        max_col = col.max()
        min_row = row.min()
        min_col = col.min()
        mid_row = torch.div(max_row + min_row, 2, rounding_mode="floor")
        mid_col = torch.div(max_col + min_col, 2, rounding_mode="floor")

        top_left = (row <= mid_row) & (col <= mid_col)
        top_right = (row <= mid_row) & (col > mid_col)
        bottom_left = (row > mid_row) & (col <= mid_col)
        bottom_right = (row > mid_row) & (col > mid_col)

        select_subset = lambda mask: Slide(
            image_path=self.image_path,
            spot_locations=self.spot_locations.select_subset(mask),
            spot_counts=self.spot_counts[mask],
            genes=self.genes
        )

        return (
            (select_subset(top_left), select_subset(~top_left)),
            (select_subset(top_right), select_subset(~top_right)),
            (select_subset(bottom_left), select_subset(~bottom_left)),
            (select_subset(bottom_right), select_subset(~bottom_right)),
        )

   # @cached_property
    def image(self):
        import PIL.Image
        import torchvision.transforms.functional as TF
        import torchvision.transforms as transforms
        if self.image_path in _image_cache:
            return _image_cache[self.image_path]

        #pil_image = PIL.Image.open(self.image_path)
        pil_image=plt.imread(self.image_path)
        #tensor_image = TF.to_tensor(pil_image)
        tensor_image = transforms.ToTensor()(np.array(pil_image))

        _image_cache[self.image_path] = tensor_image

        return tensor_image

    @staticmethod
    def load(path: str) -> 'Slide':
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)

    def save(self, path: str):
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    def log1p(self):
        return Slide(
            self.image_path,
            self.spot_locations,
            self.spot_counts.log1p(),
            self.genes,
        )
    
    def binary(self):
        return Slide(
            self.image_path,
            self.spot_locations,
            (self.spot_counts > self.spot_counts.median(dim=0)[0]).float(),
            self.genes,
        )

    def select_genes(self, genes, suppress_errors=False):
        indexes = []
        for gene in genes:
            if gene not in self.genes:
                if suppress_errors:
                    #print(f"WARNING: Gene {gene} not found in slide {self.image_path=}.")
                    continue
                
                #raise ValueError(f"Gene {gene} not found in slide {self.image_path=}.")
                raise ValueError("Gene {gene} not found in slide {self.image_path=}.")
            else:
                indexes.append(self.genes.index(gene))
        
        return Slide(
            self.image_path,
            self.spot_locations,
            self.spot_counts[:, indexes],
            genes,
        )
    
    def render(self, downsample, spot_counts, spot_size) -> PIL.Image.Image:
        """
        image: (3, height, width)
        spot_locations: SpotLocations
        spot_intensities: (num_spots,) -- intensities for *one* gene
        spot_size: int -- size of square to draw around each spot
        cmap: Maps intensity to RGB
        """
        spot_size_adjusted = torch.div(spot_size, downsample, rounding_mode="floor")
        new_image = self.image[:, ::downsample, ::downsample].clone()
        for spot in range(self.spot_locations.image_x.shape[0]):
            image_x = int((self.spot_locations.image_x[spot] - spot_size / 2) / downsample)
            image_y = int((self.spot_locations.image_y[spot] - spot_size / 2) / downsample)
            intensity = spot_counts[spot].item()
            r, g, b, a = cmap(intensity)

            gray = new_image[:, image_y:image_y + spot_size_adjusted, image_x:image_x + spot_size_adjusted].mean(dim=0)

            # (3, spot_size, spot_size)
            new_image[:, image_y:image_y + spot_size_adjusted, image_x:image_x + spot_size_adjusted] = \
                torch.stack([gray * (1 - a) + r * a, gray * (1 - a) + g * a, gray * (1 - a) + b * a])

        numpy_array = np.array(new_image.cpu().permute(1, 2, 0) * 255, dtype=np.uint8)
        pil_image = PIL.Image.fromarray(numpy_array)
        return pil_image

    @overload
    def spatialde(self) -> pd.DataFrame: ...
    @overload
    def spatialde(self, spot_counts: torch.Tensor) -> pd.DataFrame: ...
    def spatialde(self, spot_counts=None):
        import spatialde

        if spot_counts is None:
            spot_counts = self.spot_counts

        result = spatialde.run_spatialde(
            spot_counts.cpu().numpy(),
            self.genes,
            self.spot_locations.image_x.numpy(),
            self.spot_locations.image_y.numpy(),
        )

        return result

class PatchDataset(torch.utils.data.Dataset):
    def __init__(self, slide: Slide, patch_size: int, magnify: int, patch_transform, device):
        self.slide = slide
        self.patch_size = patch_size
        self.patch_transform = patch_transform
        self.device = device
        self.magnify = magnify
        self.image = self.slide.image()
    @property
    def hp(self):
        return (self.slide.hp, self.patch_size, self.magnify)

    def __getitem__(self, index: int):
        image_x, image_y = self.slide.spot_locations.image_x[index], self.slide.spot_locations.image_y[index]
        image_x = int(image_x) - self.patch_size // 2
        image_y = int(image_y) - self.patch_size // 2
        
        patch = self.slide.image()[
            :,
            image_y:image_y + self.patch_size,
            image_x:image_x + self.patch_size,
        ]
        spot_count = self.slide.spot_counts[index]
        print(image_x,image_y)
        print(image_x,image_y,self.patch_size,self.image.shape)
        if self.patch_transform is not None:
            patch = self.patch_transform(patch)

        if self.magnify != 1:
            patch = TF.resize(patch, (self.patch_size * self.magnify, self.patch_size * self.magnify))

        return patch.to(self.device), spot_count.to(self.device)

    def __len__(self):
        return self.slide.spot_locations.image_x.shape[0]

def load_spot_locations(src: str):
    import json

    with open(src) as f:
        locations = json.load(f)

    sample_spots = locations['oligo']
    sample_spots = [spot for spot in sample_spots if 'tissue' in spot and spot['tissue']]
    image_x = [spot['imageX'] for spot in sample_spots]
    image_y = [spot['imageY'] for spot in sample_spots]
    row = [spot['row'] for spot in sample_spots]
    col = [spot['col'] for spot in sample_spots]
    dia = [spot['dia'] for spot in sample_spots]

    return SpotLocations(
        image_x=torch.tensor(image_x),
        image_y=torch.tensor(image_y),
        row=torch.tensor(row),
        col=torch.tensor(col),
        dia=torch.tensor(dia),
    )

def load_spot_locations_csv(src: str):
    import json

    import pandas as pd

    locations = pd.read_csv(src, index_col='barcode')
    locations = locations.loc[locations['in_tissue'] == 1]
    barcode_order = list(locations.index)
    
    return SpotLocations(
        image_x=torch.tensor(locations['pxl_col_in_fullres'].to_numpy()),
        image_y=torch.tensor(locations['pxl_row_in_fullres'].to_numpy()),
        row=torch.tensor(locations['array_row'].to_numpy()),
        col=torch.tensor(locations['array_col'].to_numpy()),
        dia=None,
    ), barcode_order

def load_compressed_tsv(path):
    import csv
    import gzip

    return [*csv.reader(gzip.open(path, mode="rt"), delimiter="\t")]

def load_counts(matrix_path):
    import numpy as np
    import pandas as pd
    import scipy.io

    mat_filtered = scipy.io.mmread(matrix_path)

    matrix = pd.DataFrame.sparse.from_spmatrix(mat_filtered)
    dense = np.array(matrix.sparse.to_dense())

    return torch.tensor(dense).T

def cache_autostainer_slides():
    pre='../../spatial_omics/autostainer_study/input_data' 
    #pre = 'input_data/autostainer/final_data/Autostainer_20x'
    matrix_dir = pre + '/autostainer/final_data/Autostainer_20x/outs/filtered_feature_bc_matrix'

    spot_locations, barcode_order = load_spot_locations_csv(pre + '/autostainer/final_data/Autostainer_20x/outs/spatial/tissue_positions.csv')

    spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')
    barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
    spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
    spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

    genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

    slide_40x = Slide(
        image_path=pre+'/autostainer/autostain_images/_SS12254_081342_40x.tiff',
        spot_locations=spot_locations * 2,
        spot_counts=spot_counts,
        genes=genes,
    )

    slide_40x.save('autostained_extra/autostainer_40x.pkl')
def cache_autostainer_slides_2():
    pre='../../spatial_omics/autostainer_study/input_data' 
    #pre = 'input_data/autostainer/final_data/Autostainer_20x'
    matrix_dir = pre + '/autostainer/final_data/Autostainer_20x/outs/filtered_feature_bc_matrix'

    spot_locations, barcode_order = load_spot_locations_csv(pre + '/autostainer/final_data/Autostainer_20x/outs/spatial/tissue_positions.csv')

    spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')
    barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
    spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
    spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

    genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

    slide_40x = Slide(
        image_path=pre+'/preprocessed/autostainer_40x_cropped_downsampled_by_2.tif',
        spot_locations=spot_locations * 2,
        spot_counts=spot_counts,
        genes=genes,
    )

    slide_40x.save('autostained_extra/autostainer_40x_2.pkl')
    
def cache_manual_slides():
    pre = 'input_data/autostainer/final_data/Manual'
    matrix_dir = pre + '/outs/filtered_feature_bc_matrix'

    spot_locations, barcode_order = load_spot_locations_csv(pre + '/outs/spatial/tissue_positions.csv')

    spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')
    barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
    spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
    spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

    genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

    slide_20x = Slide(
        image_path='input_data/autostainer/visium_data_half_sequenced/images/EVOS/Manual_Epredia.TIF',
        spot_locations=spot_locations,
        spot_counts=spot_counts,
        genes=genes,
    )

    slide_20x.save('input_data/preprocessed/manual_20x.pkl')    

def cache_manual_slides():
    pre='../../spatial_omics/autostainer_study/input_data' 
    #pre = 'input_data/autostainer/final_data/Autostainer_20x'
    matrix_dir = pre + '/autostainer/final_data/Autostainer_20x/outs/filtered_feature_bc_matrix'

    spot_locations, barcode_order = load_spot_locations_csv(pre + '/autostainer/final_data/Autostainer_20x/outs/spatial/tissue_positions.csv')

    spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')
    barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
    spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
    spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

    genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

    slide_40x = Slide(
        image_path=pre+'/autostainer/autostain_images/_SS12254_081342_40x.tiff',
        spot_locations=spot_locations * 2,
        spot_counts=spot_counts,
        genes=genes,
    )

    slide_40x.save('autostained_extra/manual_20x.pkl')

def cache_new_slides():
    id2name = {'092534':'A_Sample1','092842':'B_Sample2','091759':'A_Sample3','092146':'B_Sample4'}
    pre='/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/autostainer_study/input_data/new_colon_visium_full_sequenced/'
    #pre = 'input_data/autostainer/final_data/Autostainer_20x'
    for uid in id2name:
        slide_name = id2name[uid]
        matrix_dir = pre + 'analysis/'+uid+'/outs/filtered_feature_bc_matrix'

        spot_locations, barcode_order = load_spot_locations_csv(pre + 'analysis/'+uid+'/outs/spatial/tissue_positions.csv')

        spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')
        barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
        spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
        spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

        genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

        slide = Slide(
            image_path=pre+'/images/Scanner/'+slide_name+'.tiff',
            spot_locations=spot_locations * 2,
            spot_counts=spot_counts,
            genes=genes,
        )

        slide.save('autostained_extra/new_seq/'+slide_name+'.pkl')
        
 
def crop_large_image_for_slide(slide: Slide, output_image_path: str):
    import tifffile
    
    # Crop the image to the area of the tissue
    xmin, xmax, ymin, ymax = slide.spot_locations.image_x.min(), slide.spot_locations.image_x.max(), slide.spot_locations.image_y.min(), slide.spot_locations.image_y.max()
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    width = xmax - xmin
    height = ymax - ymin

    print('Cropping image to', xmin, xmax, ymin, ymax, 'and adding 1024 pixels of padding')
    print('New image size will be', xmax - xmin + 2048, 'x', ymax - ymin + 2048)

    if slide.image_path.endswith('.svs'):
        image = PIL.Image.fromarray(tifffile.imread(slide.image_path)[:, :, :3])
    else:
        image = PIL.Image.open(slide.image_path)
    print(image.size)
    image = image \
        .crop((max(xmin - 1024, 0), max(ymin - 1024, 0), min(xmax + 1024, image.width), min(ymax + 1024, image.height)))
    print(image.size)
    opencv_im = np.array(image) 
    opencv_im = opencv_im[:, :, ::-1]
    cv2.imwrite(output_image_path,opencv_im)
    print('saved',output_image_path)
    #image \
    #    .crop((max(xmin - 1024, 0), max(ymin - 1024, 0), min(xmax + 1024, image.width), min(ymax + 1024, image.height))) \
     #   .save(output_image_path)

    # Adjust so (xmin - 1024, ymin - 1024) is at (0, 0)
    cropped_image_x = slide.spot_locations.image_x - (xmin - 1024)
    cropped_image_y = slide.spot_locations.image_y - (ymin - 1024)

    slide_cropped = Slide(
        image_path=output_image_path,
        spot_locations=SpotLocations(cropped_image_x, cropped_image_y, slide.spot_locations.row, slide.spot_locations.col, slide.spot_locations.dia),
        spot_counts=slide.spot_counts,
        genes=slide.genes,
    )

    return slide_cropped

    # slide_cropped.save(f'input_data/preprocessed/{slide_name}_cropped.pkl')
    
    
    
def cache_new_colon_visium_full_sequenced():
    import time
    #from test_load_large_image import crop_large_image_for_slide

    slide_ids = ['091759', '092146', '092534', '092842']
    pre1='/dartfs/rc/nosnapshots/V/VaickusL-nb/EDIT_Students/projects/spatial_omics/autostainer_study/'
    # Alignment of image names to slide ids is based on the *.sbatch files in the new_colon_visium_full_sequenced/analysis/ folder.
    image_names = ['A_Sample3', 'B_Sample4', 'A_Sample1', 'B_Sample2']

    for slide_id, image_name in zip(slide_ids, image_names):
        print("Running on slide", slide_id)

        t_start = time.time()
        input_data_path=pre1+'input_data/'
        pre = pre1+'input_data/new_colon_visium_full_sequenced/analysis/' + slide_id
        matrix_dir = pre + '/outs/filtered_feature_bc_matrix'

        # spot_locations = load_spot_locations_json('input_data/new_colon_visium_full_sequenced/images/Scanner/' + image_name + '.json')

        spot_counts = load_counts(matrix_dir + '/matrix.mtx.gz')

        print("Loaded counts @", time.time() - t_start, "seconds")

        # No reordering needed, the .json file is in the right order already
        spot_locations, barcode_order = load_spot_locations_csv(pre + '/outs/spatial/tissue_positions.csv')
        barcodes = [barcode for barcode, in load_compressed_tsv(matrix_dir + '/barcodes.tsv.gz')]
        spot_counts_by_barcode = {barcode: counts for barcode, counts in zip(barcodes, spot_counts)}
        spot_counts = torch.stack([spot_counts_by_barcode[barcode] for barcode in barcode_order], dim=0)

        spot_locations = spot_locations * 2

        print("Loaded spot locations @", time.time() - t_start, "seconds")

        genes = [gene for feature_id, gene, feature_type in load_compressed_tsv(matrix_dir + '/features.tsv.gz')]

        path = input_data_path+'/preprocessed/new_colon_visium_fs_'+slide_id+'.pkl'

        # slide_40x = Slide.load(path)

        image_path = input_data_path+'/spatial_omics_visium_v2/images/_SS12251_'+slide_id+'.svs'

        import tifffile
        
        # Crop the image to the area of the tissue
        xmin, xmax, ymin, ymax = spot_locations.image_x.min(), spot_locations.image_x.max(), spot_locations.image_y.min(), spot_locations.image_y.max()
        xmin = int(xmin)
        xmax = int(xmax)
        ymin = int(ymin)
        ymax = int(ymax)

        width = xmax - xmin
        height = ymax - ymin

        print('Cropping image to', xmin, xmax, ymin, ymax, 'and adding 1024 pixels of padding')
        print('New image size will be', xmax - xmin + 2048, 'x', ymax - ymin + 2048)

        if image_path.endswith('.svs'):
            image = PIL.Image.fromarray(tifffile.imread(image_path)[:, :, :3])
        else:
            image = PIL.Image.open(image_path)
            
        image = image.crop((max(xmin - 1024, 0), max(ymin - 1024, 0), min(xmax + 1024, image.width), min(ymax + 1024, image.height)))
        image = np.array(image)[::16, ::16, :]
        #PIL.Image.fromarray(image).save(f'{slide_id}_preview.png')

        slide_40x = crop_large_image_for_slide(Slide(
            image_path=image_path,
            spot_locations=spot_locations * 1,#2,
            spot_counts=spot_counts,
            genes=genes,
         ), 'autostained_extra/new_seq/'+ slide_id + '_crop.tiff')
        #), input_data_path+'preprocessed/new_colon_visium_40x_' + slide_id + '_crop.tiff')

        print("Done loading slide", slide_id, "@", time.time() - t_start, "seconds")

        # Save preview image
        PIL.Image.fromarray(np.array((slide_40x.image().permute(1, 2, 0) * 255)[::16, ::16, :], dtype=np.uint8)).save('autostained_extra/new_seq/new_colon_visium_40x_' + slide_id + '_preview.png')
        save_path='autostained_extra/new_seq/'+slide_id+'.pkl'
        slide_40x.save(save_path)
        print("Done saving preview image @", time.time() - t_start, "seconds")

        #exit()

    print("Done caching new_colon_visium_full_sequenced")
    
def cache_slides():
    cache_visium_slides()
    cache_autostainer_slides()
    cache_manual_slides()
