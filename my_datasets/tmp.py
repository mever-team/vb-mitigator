from datasets import load_dataset

dev_dataset = load_dataset(
    "LabHC/bias_in_bios",
    split="dev",
    cache_dir="/mnt/cephfs/home/gsarridis/projects/vb-mitigator/data/biasinbios/",
)
X_train = dev_dataset["hard_text"]
y_train = dev_dataset["profession"]
b_train = dev_dataset["gender"]
print(y_train)
