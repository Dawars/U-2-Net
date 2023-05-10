from sklearn.model_selection import train_test_split

from pathlib import Path

dataset_root = Path("/mnt/hdd/datasets/documents/Doc3D")
images = Path(dataset_root / "all.txt").read_text().split()

print(images)

train, test = train_test_split(images, test_size=0.1)

print(test)

(dataset_root / "train.txt").write_text("\n".join(train))
(dataset_root / "val.txt").write_text("\n".join(test))
