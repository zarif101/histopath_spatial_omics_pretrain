import pickle
from typing import Dict, List, Optional, Union
from typing_extensions import Literal
import os
import numpy as np
import pandas as pd
import PIL.Image
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as tfms

#from device import device
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')

DATA_DIR = '../DH/visium'

with open(DATA_DIR + '/preprocessed_data/visium_data_filtered.pkl', 'rb') as f:
    data: dict = pickle.load(f)

class SpatialOmicsSlide:
    def __init__(self,
                 array_width: int,
                 array_height: int,
                 genes: List[str],
                 patches: torch.Tensor,
                 gene_ex: torch.Tensor,
                 gene_ex_X: torch.Tensor,
                 gene_ex_Y: torch.Tensor,
                 image: np.ndarray,
                 atoi_x: Dict[int, int],
                 atoi_y: Dict[int, int],
                 average_tile_width: int,
                 average_tile_height: int,
                 ):
        self.array_width = array_width
        self.array_height = array_height
        self.genes = genes
        self.patches = patches
        self.gene_ex = gene_ex
        self.gene_ex_X = gene_ex_X
        self.gene_ex_Y = gene_ex_Y
        self.image = image
        self.atoi_x = atoi_x
        self.atoi_y = atoi_y
        self.average_tile_width = average_tile_width
        self.average_tile_height = average_tile_height

    def train_test_split(self, test_size, random_state):
        """
        Creates two new SpatialOmicsSlide objects, each with a different section of labeled gene expressions masked in
        """

        train_genes_ex, test_genes_ex, train_X, test_X, train_Y, test_Y = train_test_split(
            self.gene_ex, self.gene_ex_X, self.gene_ex_Y, test_size=test_size, random_state=random_state)

        train_slide = SpatialOmicsSlide(
            self.array_width,
            self.array_height,
            self.genes,
            self.patches,
            train_genes_ex,
            train_X,
            train_Y,
            self.image,
            self.atoi_x,
            self.atoi_y,
            self.average_tile_width,
            self.average_tile_height,
        )

        test_slide = SpatialOmicsSlide(
            self.array_width,
            self.array_height,
            self.genes,
            self.patches,
            test_genes_ex,
            test_X,
            test_Y,
            self.image,
            self.atoi_x,
            self.atoi_y,
            self.average_tile_width,
            self.average_tile_height,
        )

        return train_slide, test_slide

    def create_batches(self, batch_size, method: Union[Literal['random_sample'], Literal['random_patches']] = 'random_sample', **kwargs):
        """
        Creates batches in various methods. Returns the indices of the values to use.
         * random_sample: Selects batch_size patches at a time at random
         * batch_size: If this is 0, then it returns the whole dataset
        """

        if batch_size == 0:
            return torch.arange(self.patches.shape[0])

        if method == 'random_sample':
            index_order = torch.randperm(self.patches.shape[0])
            index_chunks = [index_order[i:i + batch_size]
                            for i in range(0, len(index_order), batch_size)]

            return index_chunks
        elif method == 'random_patches':
            square_size = kwargs['square_size']

            # Create localized patches
            for start_x in range(0, self.array_width, square_size):
                for start_y in range(0, self.array_height, square_size):
                    # Find indexes of valid genes
                    for x in range(start_x, start_x + square_size):
                        for y in range(start_y, start_y + square_size):
                            if x >= self.array_width or y >= self.array_height:
                                continue
                            yield x, y
        else:
            raise ValueError('Method not supported: {}'.format(method))

    def flip(self, direction='vertical'):
        assert direction in ['horizontal', 'vertical']

        if direction == 'vertical':
            # Image should be flipped vertically
            # atoi_y image y values and order should be flipped vertically
            #  * Prevents array y and image y from being inverse of each other, but also shifts indexes correctly
            #  * Must subtract average_tile_height to ensure that position stays in top left corner
            # Patch content and patch order should be flipped vertically
            # Gene expression indices should be flipped vertically
            new_image = self.image[::-1]
            new_atoi_y = {(len(self.atoi_y) - 1 - array_y): (
                self.image.shape[0] - image_y - self.average_tile_height) for array_y, image_y in self.atoi_y.items()}
            new_patch_content = torch.flip(self.patches, [0, 2])
            new_gene_ex_Y = len(self.atoi_y) - 1 - self.gene_ex_Y

            return SpatialOmicsSlide(
                self.array_width,
                self.array_height,
                self.genes,
                new_patch_content,
                self.gene_ex,
                self.gene_ex_X,
                new_gene_ex_Y,
                new_image,
                self.atoi_x,
                new_atoi_y,
                self.average_tile_width,
                self.average_tile_height,
            )
        else:
            # Image should be flipped horizontally
            # atoi_x image x values and order should be flipped horizontally
            #  * Prevents array x and image x from being inverse of each other, but also shifts indexes correctly
            #  * Must subtract average_tile_width to ensure that position stays in top left corner
            # Patch content and patch order should be flipped horizontally
            # Gene expression indices should be flipped horizontally
            new_image = self.image[:, ::-1]
            new_atoi_x = {(len(self.atoi_x) - 1 - array_x): (
                self.image.shape[1] - image_x - self.average_tile_width) for array_x, image_x in self.atoi_x.items()}
            new_patch_content = torch.flip(self.patches, [1, 3])
            new_gene_ex_X = len(self.atoi_x) - 1 - self.gene_ex_X

            return SpatialOmicsSlide(
                self.array_width,
                self.array_height,
                self.genes,
                new_patch_content,
                self.gene_ex,
                new_gene_ex_X,
                self.gene_ex_Y,
                new_image,
                new_atoi_x,
                self.atoi_y,
                self.average_tile_width,
                self.average_tile_height,
            )

    def get_binary_gene_ex(self):
        return (self.gene_ex > 0).type(torch.float)

    def split_into_localized_chunks(self, chunk_array_size: int = 16):
        # Naive way first
        for start_x in range(0, self.array_width - chunk_array_size, chunk_array_size):
            for start_y in range(0, self.array_height - chunk_array_size, chunk_array_size):
                try:
                    yield self._get_chunk(start_x, start_x + chunk_array_size, start_y, start_y + chunk_array_size)
                except:
                    print(
                        "Noticed a chunk without any gene expression data: {start_x=} {start_y=} {chunk_array_size=}")

    def _get_chunk(self, array_x_start, array_x_end, array_y_start, array_y_end):
        gene_ex_X = []
        gene_ex_Y = []
        gene_ex = []
        image = self.image[self.atoi_y[array_y_start]:self.atoi_y[array_y_end],
                           self.atoi_x[array_x_start]:self.atoi_x[array_x_end]]
        patches = self.patches[array_y_start:array_y_end,
                               array_x_start:array_x_end]

        for (x, y, gene_ex_) in zip(self.gene_ex_X, self.gene_ex_Y, self.gene_ex):
            if x >= array_x_start and x < array_x_end and y >= array_y_start and y < array_y_end:
                gene_ex_X.append(x - array_x_start)
                gene_ex_Y.append(y - array_y_start)
                gene_ex.append(gene_ex_)

        if len(gene_ex) == 0:
            raise Exception("No gene expression data found")

        return SpatialOmicsSlide(
            array_x_end - array_x_start,
            array_y_end - array_y_start,
            self.genes,
            patches,
            torch.stack(gene_ex),
            torch.tensor(gene_ex_X, device=device),
            torch.tensor(gene_ex_Y, device=device),
            image,
            self.atoi_x,
            self.atoi_y,
            self.average_tile_width,
            self.average_tile_height,
        )

    def apply_kmeans(self, clusters: torch.Tensor):
        # (n_patches, n_clusters)
        distance_from_each_cluster = torch.zeros(
            (self.gene_ex.shape[0], clusters.shape[0]))
        for cluster in range(clusters.shape[0]):
            distance_from_each_cluster[:, cluster] = torch.norm(
                self.gene_ex - clusters[cluster], dim=1)

        # (n_patches,)
        closest_clusters = distance_from_each_cluster.argmax(dim=-1)

        # (n_patches, n_clusters)
        new_gene_exs = torch.zeros((self.gene_ex.shape[0], clusters.shape[0]))
        new_gene_exs[range(self.gene_ex.shape[0]), closest_clusters] = 1

        return SpatialOmicsSlide(
            self.array_width,
            self.array_height,
            self.genes,
            self.patches,
            new_gene_exs,
            self.gene_ex_X,
            self.gene_ex_Y,
            self.image,
            self.atoi_x,
            self.atoi_y,
            self.average_tile_width,
            self.average_tile_height,
        )

    def apply_eigenvectors(self, vectors: torch.Tensor):
        """
        `vectors` shape: (n_vectors, n_genes)
        """
        dup = self.shallow_copy()
        dup.gene_ex = torch.mm(self.gene_ex, vectors.T)
        return dup

    def create_grid(self, patch_size: int, start_pos=(0, 0)):
        height, width, _ = self.image.shape

        new_grid_width = torch.div(
            (width - start_pos[0]), patch_size, rounding_mode='floor')
        new_grid_height = torch.div(
            (height - start_pos[1]), patch_size, rounding_mode='floor')

        xy: torch.Tensor = torch.zeros((new_grid_height * new_grid_width, 2),
                         device=device, dtype=torch.long)

        i = 0
        for x in range(start_pos[0], width - patch_size + 1, patch_size):
            for y in range(start_pos[1], height - patch_size + 1, patch_size):
                xy[i, 0] = x
                xy[i, 1] = y

                i += 1

        return xy[:i]

    def resample_grid(self, xy: torch.Tensor, patch_size: int, pooling: str):
        dup = self.shallow_copy()
        new_gene_ex = torch.zeros((xy.shape[0], self.gene_ex.shape[1]), device=device)
        i = 0
        for x, y in xy:
            # Shape (n_beads, n_genes)
            rect_ex = self.get_gene_ex_in_rect(x, x + patch_size, y, y + patch_size)
            if rect_ex.shape[0] > 0:
                # Shape (n_genes,)
                if pooling == 'median':
                    patch_expression = torch.median(rect_ex, dim=0)[0]
                elif pooling == 'mean':
                    patch_expression = torch.mean(rect_ex, dim=0)
                elif pooling == 'max':
                    patch_expression = torch.max(rect_ex, dim=0)[0]
                else:
                    raise ValueError(f"Unknown pooling method: {pooling}")

                new_gene_ex[i] = patch_expression

            i += 1
        dup.gene_ex_X = xy[:, 0]
        dup.gene_ex_Y = xy[:, 1]
        dup.gene_ex = new_gene_ex[:i]
        return dup

    def get_gene_ex_in_rect(self, start_x, end_x, start_y, end_y):
        return self.gene_ex[(self.gene_ex_X >= start_x) &
                            (self.gene_ex_X < end_x) &
                            (self.gene_ex_Y >= start_y) &
                            (self.gene_ex_Y < end_y)]

    def get_gene_ex_in_rect_with_locs(self, start_x, end_x, start_y, end_y):
        idxs = (self.gene_ex_X >= start_x) & \
                (self.gene_ex_X < end_x) & \
                (self.gene_ex_Y >= start_y) & \
                (self.gene_ex_Y < end_y)
        return self.gene_ex_X[idxs], self.gene_ex_Y[idxs], self.gene_ex[idxs]

    def with_binary_gene_ex(self):
        dup = self.shallow_copy()
        dup.gene_ex = self.get_binary_gene_ex()
        return dup

    def shallow_copy(self):
        return SpatialOmicsSlide(
            self.array_width,
            self.array_height,
            self.genes,
            self.patches,
            self.gene_ex,
            self.gene_ex_X,
            self.gene_ex_Y,
            self.image,
            self.atoi_x,
            self.atoi_y,
            self.average_tile_width,
            self.average_tile_height,
        )

    def convert_gene_ex_coordinates(self):
        dup = self.shallow_copy()
        dup.gene_ex_X = torch.zeros_like(dup.gene_ex_X)
        dup.gene_ex_Y = torch.zeros_like(dup.gene_ex_Y)

        for i in range(len(self.gene_ex)):
            dup.gene_ex_X[i] = self.atoi_x[self.gene_ex_X[i].item()]
            dup.gene_ex_Y[i] = self.atoi_y[self.gene_ex_Y[i].item()]
            x_sum += dup.gene_ex_X[i]
            y_sum += dup.gene_ex_Y[i]

        print("gene coords checksums:", x_sum, y_sum)

        return dup

    def shifted_log_transform(self):
        dup = self.shallow_copy()
        dup.gene_ex = torch.log(self.gene_ex + 1)
        return dup

    def gt_median_transform(self):
        dup = self.shallow_copy()
        median: torch.Tensor = torch.median(dup.gene_ex, dim=0)[0]
        dup.gene_ex = (dup.gene_ex > median).type(torch.int32)
        return dup

    def aggregate_clusters(self, clusters: List[List[str]], cluster_labels: List[str], aggregation: str):
        dup = self.shallow_copy()
        new_gene_ex = torch.zeros((self.gene_ex.shape[0], len(clusters)))

        genes_set = set(self.genes)
        for i, cluster in enumerate(clusters):
            indexes = [self.genes.index(gene) for gene in cluster if gene in genes_set]
            unaggregated = self.gene_ex[:, indexes]
            if aggregation == 'median':
                aggregated = torch.median(unaggregated, dim=1)[0]
            elif aggregation == 'mean':
                aggregated = torch.mean(unaggregated, dim=1)
            elif aggregation == 'max':
                aggregated = torch.max(unaggregated, dim=1)[0]
            else:
                raise ValueError("Unknown aggregation method: {aggregation}")
            new_gene_ex[:, i] = aggregated

        dup.genes = ['cluster_' + cluster_label for cluster_label in cluster_labels]
        dup.gene_ex = new_gene_ex
        return dup

    def smoothen(self, aggregation):
        new_gene_ex = torch.zeros_like(self.gene_ex)
        for i in range(self.gene_ex.shape[0]):
            x = self.gene_ex_X[i]
            y = self.gene_ex_Y[i]
            
            unaggregated = []
            for j in range(self.gene_ex.shape[0]):
                if i == j:
                    continue
                if abs(self.gene_ex_X[j] - x) < 145 and abs(self.gene_ex_Y[j] - y) < 80:
                    unaggregated.append(self.gene_ex[j])

            unaggregated = torch.stack(unaggregated)

            if aggregation == 'median':
                aggregated = torch.median(unaggregated, dim=1)[0]
            elif aggregation == 'mean':
                aggregated = torch.mean(unaggregated, dim=1)
            elif aggregation == 'max':
                aggregated = torch.max(unaggregated, dim=1)[0]

            new_gene_ex[i] = aggregated
            
        dup = self.shallow_copy()
        dup.gene_ex = new_gene_ex
        return dup

def load_training_data(id='A1', tile_size=(80, 140), verbose=False, select_genes: Optional[List[str]] = None) -> SpatialOmicsSlide:
    if id not in data:
        raise ValueError(
            "Invalid id: {id}. Valid IDs are {list(data.keys())}")

    # Check cache
    if os.path.exists(f'.cache/{id}.pkl'):
        # print("Loading slide from cache:", id)
        # slide = torch.load(f'.cache/{id}.pkl', map_location=torch.device('cpu'))
        
        # return slide
        pass

    instance = data[id]
    matrix: pd.DataFrame = instance['matrix']
    coords: pd.DataFrame = instance['coods']

    ##### Create coordinate space and coordinate gene grid with image pixels #####

    atoi_x = {}
    atoi_y = {}

    for index, row in coords.iterrows():
        barcode, in_tissue, array_row, array_col, image_row, image_col = row

        atoi_x[array_col] = image_col
        atoi_y[array_row] = image_row

    max_array_x = max(atoi_x.keys())
    max_array_y = max(atoi_y.keys())

    min_array_x = min(atoi_x.keys())
    min_array_y = min(atoi_y.keys())

    array_width = max_array_x - min_array_x + 1
    array_height = max_array_y - min_array_y + 1

    if tile_size is None:
        average_tile_width = round(
            (atoi_x[max_array_x] - atoi_x[min_array_x]) / (max_array_x - min_array_x))
        average_tile_height = round(
            (atoi_y[max_array_y] - atoi_y[min_array_y]) / (max_array_y - min_array_y))
    else:
        average_tile_width, average_tile_height = tile_size

    if verbose:
        print(f"Dataset {id}")
        print(
            "Average tile size: ({average_tile_width}, {average_tile_height})")
        print(
            "(x, y): ({min_array_x}, {min_array_y}) -> ({max_array_x}, {max_array_y})")
        print("Gene count:", len(matrix))
        print("Sample count:", len(coords))
        print()

    # Unfortunately, gene expression data patches are not of constant size
    # So we must do some torch transforms

    ##### Splitting image into patches #####
    patches = torch.zeros((array_height, array_width, 3,
                          average_tile_height, average_tile_width), device=device)
    image = np.array(PIL.Image.open(DATA_DIR + '/raw_data/' + id + '.TIF'))

    transform = tfms.Compose([
        tfms.ToTensor(),
        tfms.Resize((average_tile_height, average_tile_width)),
    ])

    for x in range(min_array_x, max_array_x + 1):
        image_x = atoi_x[x]
        if (x + 1) in atoi_x:
            next_image_x = atoi_x[x + 1]
        else:
            next_image_x = image_x + average_tile_width

        for y in range(min_array_y, max_array_y + 1):
            image_y = atoi_y[y]
            if (y + 1) in atoi_y:
                next_image_y = atoi_y[y + 1]
            else:
                next_image_y = image_y + average_tile_height

            patches[y][x] = transform(
                image[image_y:next_image_y, image_x:next_image_x, :])

    # first three columns are feature_type, gene, feature_id
    barcodes: list = matrix.columns[3:].to_list()
    genes = matrix['gene'].to_list()
    gene_to_barcode: np.ndarray = matrix[barcodes].to_numpy().astype(np.int32)
    barcode_to_gene = gene_to_barcode.T
    num_genes, num_barcodes = gene_to_barcode.shape

    if select_genes is not None:
        select_gene_indexes = [genes.index(gene) for gene in select_genes]
        num_genes = len(select_genes)
    else:
        select_gene_indexes = None

    ##### CORRELATING TILES TO GENE EXPRESSIONS #####
    # We do not have gene expressions for every tile. Therefore, we must record the X and Y values of each tile
    gene_ex_X = torch.zeros(
        (num_barcodes), dtype=torch.long, device=device)
    gene_ex_Y = torch.zeros(
        (num_barcodes), dtype=torch.long, device=device)
    gene_ex = torch.zeros(
        (num_barcodes, num_genes), dtype=torch.float32, device=device)

    i_barcode = 0
    for _, row in coords.iterrows():
        barcode, in_tissue, array_y, array_x, image_y, image_x = row

        # It's possible for a barcode to exist in coords without having a measured gene expression (idk why tho)
        if barcode in barcodes and in_tissue:
            barcode_index = barcodes.index(barcode)
            gene_expression_for_barcode = barcode_to_gene[barcode_index]
            gene_ex_X[i_barcode] = image_x  # array_x
            gene_ex_Y[i_barcode] = image_y  # array_y
            if select_gene_indexes is not None:
                gene_ex[i_barcode] = torch.from_numpy(
                    gene_expression_for_barcode)[select_gene_indexes]
            else:
                gene_ex[i_barcode] = torch.from_numpy(
                    gene_expression_for_barcode)

            i_barcode += 1

    gene_ex = gene_ex[:i_barcode]
    gene_ex_X = gene_ex_X[:i_barcode]
    gene_ex_Y = gene_ex_Y[:i_barcode]

    slide = SpatialOmicsSlide(
        array_width,
        array_height,
        genes,
        patches,
        gene_ex,
        gene_ex_X,
        gene_ex_Y,
        image,
        atoi_x,
        atoi_y,
        average_tile_width,
        average_tile_height,
    )

    if not os.path.exists(f'.cache/{id}.pt'):
        # Cache
        if not os.path.exists('.cache'):
            os.mkdir('.cache')
        
        torch.save(slide, f'.cache/{id}.pt')

    return slide

def print_dataset_information():
    # Second arg is 'verbose'
    a1 = load_training_data('A1', verbose=True)
    b1 = load_training_data('B1', verbose=True)
    c1 = load_training_data('C1', verbose=True)
    d1 = load_training_data('D1', verbose=True)

def create_dataframe(input_roc_files: List[str], input_roc_titles: List[str]):
    import pandas as pd

    genes = load_training_data('A1').genes

    data = {
        f'{title}_roc': np.load(input_roc_file)
        for input_roc_file, title in zip(input_roc_files, input_roc_titles)
    }

    df = pd.DataFrame(data, index=genes)
    df.to_csv('roc_data.csv')

def load_unlabeled_slide(path, patch_size: int, device: torch.device) -> SpatialOmicsSlide:
    print('Loading unlabeled slide')

    image: np.ndarray = np.load(path)

    tiles_width = image.shape[1] // patch_size
    tiles_height = image.shape[0] // patch_size

    print('Tiling slide')

    # See https://discuss.pytorch.org/t/creating-nonoverlapping-patches-from-3d-data-and-reshape-them-back-to-the-image/51210/6
    
    # Permutes order
    patches = torch.zeros((tiles_height, tiles_width, image.shape[2], patch_size, patch_size), device=device, dtype=torch.float32)
    for i in range(tiles_width):
        for j in range(tiles_height):
            patches[j, i] = torch.from_numpy(
                image[j * patch_size:(j + 1) * patch_size, i * patch_size:(i + 1) * patch_size]
            ).permute(2, 0, 1).to(device) / 256

    print('Constructing SpatialOmicsSlide')

    return SpatialOmicsSlide(
        tiles_width,
        tiles_height,
        [],
        patches,
        None,
        None,
        None,
        image,
        {},
        {},
        patch_size,
        patch_size,
    )

if __name__ == "__main__":
    create_dataframe([
        './out/ViT_n=2_batchsize=256_D1_500/roc_auc_scores.npy',
        './out/ViT_n=2_batchsize=256/roc_auc_scores_test_500.npy',
        './out/ViT_n=2_batchsize=256/roc_auc_scores_train_500.npy',
    ], [
        'different_slide_validation__Test',
        'same_slide_validation__Test',
        'same_slide_validation__Train',
    ])
