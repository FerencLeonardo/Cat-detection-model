import random
import shutil
from pathlib import Path
from pycocotools.coco import COCO

# Setup paths.
root = Path("C:/Users/leoli/Desktop/Projects/Data/")
in_root = root / "Input data" / "Coco dataset 2017" 
out_root = root / "Output data"
annotation_file = in_root / "annotations" / "instances_train2017.json"

# Load the dataset with COCO from pycocotools.
# Generates dictionaries for the data and allows fast querying.
coco = COCO(str(annotation_file))

# Creates a list with category ID corresponding to 'cat'.
# Get category ID list for the label 'cat' (only element will be the 'cat' category ID).
cat_id = coco.getCatIds(catNms=['cat'])

# Get all images' IDs.
all_img_ids = coco.getImgIds()

# Get all images' IDs that have the 'cat' category ID.
cat_img_ids = coco.getImgIds(catIds=cat_id)

# Subtract set of 'cat' image IDs from set of all image IDs and convert resulting set into a list. 
non_cat_img_ids = list(set(all_img_ids) - set(cat_img_ids))

SAMPLE_SIZE = 3000

# Raise error if there are less images in the dataset than the sample size.
if len(cat_img_ids) < SAMPLE_SIZE:
    raise ValueError(f"Only {len(cat_img_ids)} cat images available, need at least {SAMPLE_SIZE}")

if len(non_cat_img_ids) < SAMPLE_SIZE:
    raise ValueError(f"Only {len(non_cat_img_ids)} non-cat images available, need at least {SAMPLE_SIZE}")

# Get the randomly selected 'cat' image IDs.
sample_cat_img_ids = random.sample(cat_img_ids, SAMPLE_SIZE)

# Get the randomly selected non-cat image IDs.
sample_non_cat_img_ids = random.sample(non_cat_img_ids, SAMPLE_SIZE)