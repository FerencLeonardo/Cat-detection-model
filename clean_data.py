import random, shutil
from pathlib import Path
from pycocotools.coco import COCO

class CleanData:
    def __init__(self, root="C:/Users/leoli/Desktop/Projects/Data/", ann_rel ="Input data/Coco dataset 2017/annotations/instances_train2017.json", imgs_rel ="Input data/Coco dataset 2017/train2017",):
        self.root = Path(root)

        self.imgs_dir = self.root / imgs_rel # Path where the image files are stored.

        self.ann_path = self.root / ann_rel # Path where the annotations are stored.
        self.coco = COCO(str(self.ann_path))

    def _check_splits(self, splits):
        if len(splits) != 3 or round(sum(splits)) != 1:
            raise ValueError("splits must be [train, val, test] and sum to 1.0, e.g. [0.70, 0.15, 0.15]")

    def _make_out_dirs(self, out_root):
        for split in ["train", "val", "test"]:
            for kind in ["positives", "negatives"]:
                (out_root / split / kind).mkdir(parents=True, exist_ok=True)

    # Take sample size, and multiple by splits' ratios to get ammount of images from the samples in each split.
    def _split_counts(self, n, splits):
        n_train = int(n * splits[0])
        n_val = int(n * splits[1])
        return n_train, n_val

    def _copy_by_ids(self, ids, split, kind, out_root):
        for id in ids:
            info = self.coco.loadImgs([id])[0] # Get dictionary with metadata of image with current ID.
            src = self.imgs_dir / info["file_name"] # Get the source path for the image.
            dst = out_root / split / kind / info["file_name"] # Set the destination path for the image to be copied.
            if not dst.exists():
                shutil.copy2(src, dst) # If image doesn't already exist in the output folder, then copy it to the destionation path.

    def clean_data(self, label=['cat'], splits=[0.70, 0.15, 0.15], sample_size=3000, seed=0):
        self._check_splits(splits)
        random.seed(seed)

        lb = label[0]                   
        out_root = self.root / "Output data" / lb
        self._make_out_dirs(out_root)

        # Get category ID.
        pos_cat_ids = self.coco.getCatIds(catNms=label)     

        # Get image IDs.
        all_img_ids = self.coco.getImgIds()
        pos_img_ids = sorted(self.coco.getImgIds(catIds=pos_cat_ids))
        neg_img_ids = sorted(list(set(all_img_ids) - set(pos_img_ids)))

        if len(pos_img_ids) < sample_size:
            raise ValueError(f"Only {len(pos_img_ids)} {lb} images available, need at least {sample_size}")
        if len(neg_img_ids) < sample_size:
            raise ValueError(f"Only {len(neg_img_ids)} not-{lb} images available, need at least {sample_size}")

        # Pick a sample size ammount of images.
        pos_sample = random.sample(pos_img_ids, sample_size)
        neg_sample = random.sample(neg_img_ids, sample_size)

        # Shuffle to make it unbiased.
        random.shuffle(pos_sample)
        random.shuffle(neg_sample)

        # Split image IDs into train, val, test.
        n_train, n_val = self._split_counts(sample_size, splits)
        n_sum = n_train + n_val
        pos_train, pos_val, pos_test = pos_sample[:n_train], pos_sample[n_train:n_sum], pos_sample[n_sum:]
        neg_train, neg_val, neg_test = neg_sample[:n_train], neg_sample[n_train:n_sum], neg_sample[n_sum:]


        # Copy images to output directories
        self._copy_by_ids(pos_train, "train", "positives", out_root)
        self._copy_by_ids(pos_val,   "val",   "positives", out_root)
        self._copy_by_ids(pos_test,  "test",  "positives", out_root)

        self._copy_by_ids(neg_train, "train", "negatives", out_root)
        self._copy_by_ids(neg_val,   "val",   "negatives", out_root)
        self._copy_by_ids(neg_test,  "test",  "negatives", out_root)

        print("Images copied!")

def main():
    cleaner = CleanData()
    cleaner.clean_data()

if __name__ == "__main__":
    main()
