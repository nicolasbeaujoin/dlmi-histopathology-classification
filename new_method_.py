"""
Improved baseline for histopathology OOD classification.

Requirements:
  pip install torch torchvision timm huggingface_hub h5py tqdm pandas lightgbm
"""

import os
import h5py
import torch
import random
import timm
import numpy as np
import pandas as pd
import lightgbm as lgb

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring


# ===========================================================================
# Configuration
# ===========================================================================

TRAIN_IMAGES_PATH = "datasets/train.h5"
VAL_IMAGES_PATH = "datasets/val.h5"
TEST_IMAGES_PATH = "datasets/test.h5"
SEED = 0

BATCH_SIZE = 32
FEATURES_DIR = "features_uni2h"
os.makedirs(FEATURES_DIR, exist_ok=True)

torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)


# ===========================================================================
# 1. Dataset
# ===========================================================================


class BaselineDataset(Dataset):
    def __init__(self, dataset_path, preprocessing, mode):
        super(BaselineDataset, self).__init__()
        self.dataset_path = dataset_path
        self.preprocessing = preprocessing
        self.mode = mode

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, "r") as hdf:
            img = torch.tensor(hdf.get(img_id).get("img"))
            label = (
                np.array(hdf.get(img_id).get("label"))
                if self.mode == "train"
                else self.image_ids[idx]
            )
        return self.preprocessing(img).float(), label


class CONCHDataset(Dataset):
    """
    Dataset for CONCH: returns PIL-preprocessed images.
    CONCH's preprocess expects PIL images, not tensors, so we
    convert the h5 tensor [C,H,W] → PIL → preprocess.
    """

    def __init__(self, dataset_path, preprocess, mode):
        super().__init__()
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        self.mode = mode

        with h5py.File(self.dataset_path, "r") as hdf:
            self.image_ids = list(hdf.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        from torchvision.transforms.functional import to_pil_image

        img_id = self.image_ids[idx]
        with h5py.File(self.dataset_path, "r") as hdf:
            img = torch.tensor(hdf.get(img_id).get("img"))
            label = (
                np.array(hdf.get(img_id).get("label"))
                if self.mode == "train"
                else self.image_ids[idx]
            )
        # CONCH preprocess expects PIL image
        pil_img = to_pil_image(img)
        return self.preprocess(pil_img), label


# ===========================================================================
# 2. Load UNI2-h
# ===========================================================================


def load_uni2h(device):
    """
    Load UNI2-h (ViT-H/14, 1536-dim) from HuggingFace.
    Requires prior access approval at https://huggingface.co/MahmoodLab/UNI2-h
    """
    print("Loading UNI2-h model...")
    timm_kwargs = {
        "img_size": 224,
        "patch_size": 14,
        "depth": 24,
        "num_heads": 24,
        "init_values": 1e-5,
        "embed_dim": 1536,
        "mlp_ratio": 2.66667 * 2,
        "num_classes": 0,
        "no_embed_class": True,
        "mlp_layer": timm.layers.SwiGLUPacked,
        "act_layer": torch.nn.SiLU,
        "reg_tokens": 8,
        "dynamic_img_size": True,
    }
    model = timm.create_model(
        "hf-hub:MahmoodLab/UNI2-h", pretrained=True, **timm_kwargs
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model = model.to(device)
    model.eval()
    num_features = model.embed_dim  # 1536
    print(f"UNI2-h loaded — embedding dim: {num_features}")
    return model, transform, num_features


def load_conch(device):
    """
    Load CONCH (ViT-B/16, 512-dim) from HuggingFace.
    Install first: pip install git+https://github.com/mahmoodlab/CONCH.git
    Requires access approval at https://huggingface.co/MahmoodLab/CONCH
    """
    print("Loading CONCH model...")
    from conch.open_clip_custom import create_model_from_pretrained

    model, preprocess = create_model_from_pretrained(
        "conch_ViT-B-16", "hf_hub:MahmoodLab/conch"
    )
    model = model.to(device)
    model.eval()
    print("CONCH loaded — embedding dim: 512")
    return model, preprocess


def load_virchow2(device):
    """
    Load Virchow2 (ViT-H/14, 1280-dim CLS token, 2560-dim with pooled patches).
    Requires access approval at https://huggingface.co/paige-ai/Virchow2
    """
    print("Loading Virchow2 model...")
    model = timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=True,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    )
    transform = create_transform(
        **resolve_data_config(model.pretrained_cfg, model=model)
    )
    model = model.to(device)
    model.eval()
    print("Virchow2 loaded — embedding dim: 2560 (1280 CLS + 1280 pooled patches)")
    return model, transform


# ===========================================================================
# 3. Feature extraction and caching
# ===========================================================================


def extract_and_save_uni2h(model, dataloader, device, split_name):
    """
    Extract features with fp16, save to disk, return tensors.
    Skips extraction if cached files already exist.
    """
    feat_path = os.path.join(FEATURES_DIR, f"{split_name}_features.pt")
    label_path = os.path.join(FEATURES_DIR, f"{split_name}_labels.pt")

    if os.path.exists(feat_path):
        print(f"  Loading cached {split_name} features from {FEATURES_DIR}/")
        features = torch.load(feat_path)
        labels = torch.load(label_path)
        return features, labels

    print(f"  Extracting {split_name} features...")
    all_features = []
    all_labels = []
    with torch.no_grad(), torch.autocast(device_type="cuda"):
        for batch_imgs, batch_labels in tqdm(
            dataloader, desc=f"  {split_name}", leave=False
        ):
            feats = model(batch_imgs.to(device)).cpu()
            all_features.append(feats)
            if isinstance(batch_labels, (list, tuple)):
                all_labels.extend(batch_labels)
            else:
                all_labels.append(batch_labels)

    features = torch.cat(all_features, dim=0)
    try:
        labels = torch.cat(all_labels, dim=0)
    except TypeError:
        # Test mode: labels are string IDs, keep as list
        labels = all_labels

    torch.save(features, feat_path)
    torch.save(labels, label_path)
    print(f"  Saved {split_name} features: {features.shape}")
    return features, labels


def extract_and_save_conch(model, dataloader, device, split_name, features_dir):
    """
    Extract CONCH features using model.encode_image().
    Uses proj_contrast=False, normalize=False for linear probing.
    """
    feat_path = os.path.join(features_dir, f"{split_name}_features.pt")
    label_path = os.path.join(features_dir, f"{split_name}_labels.pt")

    if os.path.exists(feat_path):
        print(f"  Loading cached {split_name} features from {features_dir}/")
        features = torch.load(feat_path)
        labels = torch.load(label_path)
        return features, labels

    print(f"  Extracting {split_name} CONCH features...")
    all_features = []
    all_labels = []
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for batch_imgs, batch_labels in tqdm(
            dataloader, desc=f"  {split_name}", leave=False
        ):
            feats = model.encode_image(
                batch_imgs.to(device), proj_contrast=False, normalize=False
            ).cpu()
            all_features.append(feats)
            if isinstance(batch_labels, (list, tuple)):
                all_labels.extend(batch_labels)
            else:
                all_labels.append(batch_labels)

    features = torch.cat(all_features, dim=0)
    try:
        labels = torch.cat(all_labels, dim=0)
    except TypeError:
        labels = all_labels

    torch.save(features, feat_path)
    torch.save(labels, label_path)
    print(f"  Saved {split_name} features: {features.shape}")
    return features, labels


def extract_and_save_virchow2(model, dataloader, device, split_name, features_dir):
    """
    Extract Virchow2 features.
    Virchow2 returns all tokens [B, 257, 1280] (CLS + 256 patches).
    We concatenate CLS token with mean-pooled patch tokens → 2560-dim,
    as recommended by the official documentation.
    """
    feat_path = os.path.join(features_dir, f"{split_name}_features.pt")
    label_path = os.path.join(features_dir, f"{split_name}_labels.pt")

    if os.path.exists(feat_path):
        print(f"  Loading cached {split_name} features from {features_dir}/")
        features = torch.load(feat_path)
        labels = torch.load(label_path)
        return features, labels

    print(f"  Extracting {split_name} Virchow2 features...")
    all_features = []
    all_labels = []
    with torch.inference_mode(), torch.autocast(device_type="cuda"):
        for batch_imgs, batch_labels in tqdm(
            dataloader, desc=f"  {split_name}", leave=False
        ):
            output = model(batch_imgs.to(device))  # [B, 257, 1280]
            class_token = output[:, 0]  # [B, 1280]
            patch_tokens = output[:, 5:]  # [B, 256, 1280]
            # Concatenate CLS + mean-pooled patches → [B, 2560]
            embedding = (
                torch.cat([class_token, patch_tokens.mean(1)], dim=-1).float().cpu()
            )
            all_features.append(embedding)
            if isinstance(batch_labels, (list, tuple)):
                all_labels.extend(batch_labels)
            else:
                all_labels.append(batch_labels)

    features = torch.cat(all_features, dim=0)
    try:
        labels = torch.cat(all_labels, dim=0)
    except TypeError:
        labels = all_labels

    torch.save(features, feat_path)
    torch.save(labels, label_path)
    print(f"  Saved {split_name} features: {features.shape}")
    return features, labels


def load_features(features_dir, split):
    """
    Load precomputed features and labels.

    Args:
        features_dir: path to the features folder (e.g. "features_uni2h")
        split: one of "train", "val", "test"
    Returns:
        features (tensor), labels (tensor or list for test)
    """
    features = torch.load(os.path.join(features_dir, f"{split}_features.pt"))
    labels = torch.load(os.path.join(features_dir, f"{split}_labels.pt"))
    return features, labels


# ===========================================================================
# 4. Main
# ===========================================================================


def get_uni2h_embeddings():
    # from huggingface_hub import login

    # login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load UNI2-h ---
    feature_extractor, uni_transform, num_features = load_uni2h(device)

    train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, uni_transform, "train")
    val_dataset = BaselineDataset(VAL_IMAGES_PATH, uni_transform, "train")
    test_dataset = BaselineDataset(TEST_IMAGES_PATH, uni_transform, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # --- Extract features ---
    print("\n=== Extracting features ===")
    train_x, train_y = extract_and_save_uni2h(
        feature_extractor, train_loader, device, "train"
    )
    val_x, val_y = extract_and_save_uni2h(feature_extractor, val_loader, device, "val")
    test_x, test_y = extract_and_save_uni2h(
        feature_extractor, test_loader, device, "test"
    )

    print(f"Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
    print("UNI2-h feature extraction done!")


def get_conch_embeddings():
    # from huggingface_hub import login

    # login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    conch_features_dir = "features_conch"
    os.makedirs(conch_features_dir, exist_ok=True)

    # --- Load CONCH ---
    model, preprocess = load_conch(device)

    train_dataset = CONCHDataset(TRAIN_IMAGES_PATH, preprocess, "train")
    val_dataset = CONCHDataset(VAL_IMAGES_PATH, preprocess, "train")
    test_dataset = CONCHDataset(TEST_IMAGES_PATH, preprocess, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # --- Extract features ---
    print("\n=== Extracting CONCH features ===")
    train_x, train_y = extract_and_save_conch(
        model, train_loader, device, "train", conch_features_dir
    )
    val_x, val_y = extract_and_save_conch(
        model, val_loader, device, "val", conch_features_dir
    )
    test_x, test_y = extract_and_save_conch(
        model, test_loader, device, "test", conch_features_dir
    )

    print(f"Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
    print("CONCH feature extraction done!")


def get_virchow2_embeddings():
    # from huggingface_hub import login

    # login()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    virchow2_features_dir = "features_virchow2"
    os.makedirs(virchow2_features_dir, exist_ok=True)

    # --- Load Virchow2 ---
    model, transform = load_virchow2(device)

    # Uses BaselineDataset — same tensor pipeline as UNI2-h
    train_dataset = BaselineDataset(TRAIN_IMAGES_PATH, transform, "train")
    val_dataset = BaselineDataset(VAL_IMAGES_PATH, transform, "train")
    test_dataset = BaselineDataset(TEST_IMAGES_PATH, transform, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    # --- Extract features ---
    print("\n=== Extracting Virchow2 features ===")
    train_x, train_y = extract_and_save_virchow2(
        model, train_loader, device, "train", virchow2_features_dir
    )
    val_x, val_y = extract_and_save_virchow2(
        model, val_loader, device, "val", virchow2_features_dir
    )
    test_x, test_y = extract_and_save_virchow2(
        model, test_loader, device, "test", virchow2_features_dir
    )

    print(f"Train: {train_x.shape}, Val: {val_x.shape}, Test: {test_x.shape}")
    print("Virchow2 feature extraction done!")


def run(predict=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_x, train_y = load_features("features_virchow2", "train")
    val_x, val_y = load_features("features_virchow2", "val")
    test_x, test_y = load_features("features_virchow2", "test")

    train_x_uni, _ = load_features("features_uni2h", "train")
    val_x_uni, _ = load_features("features_uni2h", "val")
    test_x_uni, _ = load_features("features_uni2h", "test")

    combined_train = torch.cat([train_x_uni, train_x], dim=1)
    print(combined_train.shape)
    combined_val = torch.cat([val_x_uni, val_x], dim=1)
    print(combined_val.shape)
    combined_test = torch.cat([test_x_uni, test_x], dim=1)
    print(combined_test.shape)

    # print(f"Shape: {train_x.shape}")
    # print(f"Mean:  {train_x.mean():.4f}")
    # print(f"Std:   {train_x.std():.4f}")
    # print(f"Min:   {train_x.min():.4f}")
    # print(f"Max:   {train_x.max():.4f}")
    # print(f"Any NaN: {train_x.isnan().any()}")

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=1e-3, max_iter=1000)),
        ]
    )
    clf.fit(combined_train.numpy(), train_y.numpy())
    print(f"Val accuracy: {clf.score(combined_val.numpy(), val_y.numpy()):.4f}")

    # torch.random.manual_seed(SEED)

    # mlp = NeuralNetClassifier(
    #     torch.nn.Linear(4096, 2).to(device),
    #     max_epochs=20,
    #     lr=5e-4,
    #     optimizer=torch.optim.SGD,
    #     optimizer__momentum=0.9,
    #     optimizer__weight_decay=5e-4,
    #     # optimizer__weight_decay=0,
    #     criterion=torch.nn.CrossEntropyLoss,
    #     iterator_train__shuffle=True,
    #     iterator_train__batch_size=8,
    #     callbacks=[EpochScoring(scoring="accuracy", lower_is_better=False)],
    # )

    # mlp.fit(combined_train, train_y)
    # mlp.fit(train_x, train_y)

    # Evaluate on validation set
    # val_accuracy = mlp.score(combined_val, val_y)
    # val_accuracy = mlp.score(val_x, val_y)
    # print(f"Validation accuracy: {val_accuracy:.4f}")

    if predict:
        # test_pred = mlp.predict(combined_test.to(device))
        # test_pred = mlp.predict(test_x.to(device))
        test_pred = clf.predict(combined_test.numpy())

        # test_y contains the image IDs (strings from h5 keys) for test mode
        df = pd.DataFrame({"ID": test_y, "Pred": test_pred}).set_index("ID")
        df.to_csv("submission_virchow2_uni2h_lr.csv")
        print(f"Saved {len(df)} predictions")


if __name__ == "__main__":
    run(predict=False)
