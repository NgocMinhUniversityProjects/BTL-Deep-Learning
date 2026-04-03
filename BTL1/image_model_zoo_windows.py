from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# Keep BLAS thread fan-out small across DataLoader worker processes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import timm

    HAS_TIMM = True
except Exception:
    HAS_TIMM = False

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
    from lightning.pytorch.loggers import CSVLogger as PLCSVLogger
    from lightning.pytorch.loggers import WandbLogger as PLWandbLogger

    HAS_LIGHTNING = True
except Exception:
    try:
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
        from pytorch_lightning.loggers import CSVLogger as PLCSVLogger
        from pytorch_lightning.loggers import WandbLogger as PLWandbLogger

        HAS_LIGHTNING = True
    except Exception:
        HAS_LIGHTNING = False

try:
    from torch.utils.tensorboard import SummaryWriter  # noqa: F401

    HAS_TENSORBOARD = True
except Exception:
    HAS_TENSORBOARD = False

try:
    import wandb

    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

try:
    from lightning.pytorch.loggers import LitLogger as PLLitLogger

    HAS_LITLOGGER = True
except Exception:
    HAS_LITLOGGER = False

try:
    from lightning.pytorch.utilities.warnings import PossibleUserWarning
except Exception:
    PossibleUserWarning = UserWarning


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Notebook-parity runner for BTL1 image model zoo on Windows")
    p.add_argument("--dataset-source", default="misrakahmed/vegetable-image-dataset")
    p.add_argument("--dataset-key", default="vegetable_images")
    p.add_argument("--download-path", default="./data/image")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=int(os.getenv("NUM_WORKERS", "4")))
    p.add_argument("--finetune-modes", default="scratch,head_only,two_stage")
    p.add_argument("--finetune-mode", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--full-model-set", action="store_true", default=True)
    p.add_argument("--light-model-set", action="store_true")
    p.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "image-classification"))
    p.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    p.add_argument("--wandb-mode", default=os.getenv("WANDB_MODE", "online"))
    p.add_argument("--wandb-init-timeout", type=int, default=int(os.getenv("WANDB_INIT_TIMEOUT", "180")))
    p.add_argument("--wandb-auto-sync", action="store_true", default=os.getenv("WANDB_AUTO_SYNC_MINISTEP", "1") == "1")
    p.add_argument("--wandb-sync-every", type=int, default=int(os.getenv("WANDB_SYNC_EVERY_N_MINISTEPS", "1")))
    p.add_argument("--wandb-sync-timeout", type=int, default=int(os.getenv("WANDB_SYNC_TIMEOUT", "120")))
    p.add_argument("--wandb-recover-online", action="store_true", default=os.getenv("WANDB_RECOVER_ONLINE", "1") == "1")
    p.add_argument("--use-litlogger", action="store_true", default=os.getenv("USE_LITLOGGER", "0") == "1")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_finetune_modes(modes_text: str, single_mode: str | None) -> list[str]:
    if single_mode:
        modes = [single_mode.strip()]
    else:
        modes = [m.strip() for m in str(modes_text).split(",") if m.strip()]

    if not modes:
        raise ValueError("At least one finetune mode must be provided.")

    valid_modes = {"scratch", "head_only", "two_stage"}
    invalid = [m for m in modes if m not in valid_modes]
    if invalid:
        raise ValueError(f"Unsupported finetune mode(s): {invalid}. Valid: {sorted(valid_modes)}")
    return modes


args = parse_args()

SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent


def resolve_data_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()

# Dataset acquisition config
DATASET_CONFIG = {
    "dataset_source": args.dataset_source,
    "dataset_key": args.dataset_key,
    "download_path": str(resolve_data_path(args.download_path)),
}
dataset_name = DATASET_CONFIG["dataset_source"]
dataset_key = DATASET_CONFIG["dataset_key"]
download_path = DATASET_CONFIG["download_path"]

# Runtime and training config
SEED = int(args.seed)
IMG_SIZE = 224
BATCH_SIZE = int(args.batch_size)
NUM_WORKERS = max(0, int(args.num_workers))
EPOCHS = int(args.epochs)
FREEZE_EPOCHS = 2
FINETUNE_MODES = parse_finetune_modes(args.finetune_modes, args.finetune_mode)
RUN_FULL_MODEL_SET = not args.light_model_set

USE_CSV_TRAINLOG = True
USE_TENSORBOARD = False
USE_WANDB = (not args.no_wandb) and HAS_WANDB
WANDB_PROJECT = args.wandb_project
WANDB_ENTITY = args.wandb_entity
WANDB_MODE = args.wandb_mode
WANDB_INIT_TIMEOUT = int(args.wandb_init_timeout)
WANDB_AUTO_SYNC_MINISTEP = bool(args.wandb_auto_sync)
WANDB_SYNC_EVERY_N_MINISTEPS = int(args.wandb_sync_every)
WANDB_SYNC_TIMEOUT = int(args.wandb_sync_timeout)
WANDB_RECOVER_ONLINE = bool(args.wandb_recover_online)
USE_LITLOGGER = bool(args.use_litlogger)

USE_AMP = True
USE_CHANNELS_LAST = True
USE_TORCH_COMPILE = False
PREFETCH_FACTOR = 4
RESUME_FROM_CHECKPOINT = not args.no_resume
SAVE_CHECKPOINT_EVERY_EPOCH = True

PIPELINE_NAME = "image"
dataset_tag = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(dataset_key or dataset_name or "vegetable_images"))
LOG_DIR = SCRIPT_DIR / "log" / PIPELINE_NAME / dataset_tag
CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint" / PIPELINE_NAME / dataset_tag
RESULT_DIR = SCRIPT_DIR / "results" / PIPELINE_NAME
STATE_DIR = CHECKPOINT_DIR / "state"
PIPELINE_STATE_FILE = STATE_DIR / "pipeline_state.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_json_bytes(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode("utf-8")


def build_config_fingerprint(extra=None) -> str:
    base = {
        "dataset_tag": dataset_tag,
        "dataset_name": str(dataset_name),
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "freeze_epochs": FREEZE_EPOCHS,
        "seed": SEED,
        "use_amp": USE_AMP,
        "use_channels_last": USE_CHANNELS_LAST,
        "use_torch_compile": USE_TORCH_COMPILE,
    }
    if extra is not None:
        base["extra"] = extra
    return hashlib.sha256(stable_json_bytes(base)).hexdigest()


GLOBAL_CONFIG_FINGERPRINT = build_config_fingerprint()


def state_path(name: str) -> Path:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    return STATE_DIR / name


def load_pipeline_state() -> dict:
    if not PIPELINE_STATE_FILE.exists():
        return {
            "dataset_tag": dataset_tag,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "steps": {},
        }
    return json.loads(PIPELINE_STATE_FILE.read_text(encoding="utf-8"))


def save_pipeline_state(state: dict) -> None:
    state["updated_at"] = now_iso()
    PIPELINE_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def update_pipeline_step(step: str, status: str, fingerprint=None, artifacts=None, detail=None) -> None:
    state = load_pipeline_state()
    steps = state.setdefault("steps", {})
    current = steps.get(step, {})
    current.update(
        {
            "status": status,
            "fingerprint": fingerprint,
            "artifacts": artifacts or current.get("artifacts", {}),
            "detail": detail or current.get("detail", {}),
            "updated_at": now_iso(),
        }
    )
    if "created_at" not in current:
        current["created_at"] = now_iso()
    steps[step] = current
    save_pipeline_state(state)


def step_is_completed(step: str, fingerprint=None, required_files=None) -> bool:
    state = load_pipeline_state()
    info = state.get("steps", {}).get(step)
    if not info or info.get("status") != "completed":
        return False
    if fingerprint is not None and info.get("fingerprint") != fingerprint:
        return False
    if required_files:
        return all(Path(p).exists() for p in required_files)
    return True


def save_split_state(train_df, val_df, test_df, label_to_idx, idx_to_label) -> None:
    train_path = state_path("train_split.csv")
    val_path = state_path("val_split.csv")
    test_path = state_path("test_split.csv")
    meta_path = state_path("meta.json")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    split_hash = hashlib.sha256(
        stable_json_bytes(
            {
                "train": int(len(train_df)),
                "val": int(len(val_df)),
                "test": int(len(test_df)),
                "label_to_idx": label_to_idx,
            }
        )
    ).hexdigest()
    meta = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "num_classes": len(label_to_idx),
        "split_hash": split_hash,
    }
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    update_pipeline_step(
        "split_state",
        "completed",
        fingerprint=build_config_fingerprint({"step": "split_state"}),
        artifacts={
            "train_split": str(train_path),
            "val_split": str(val_path),
            "test_split": str(test_path),
            "meta": str(meta_path),
        },
        detail={"split_hash": split_hash},
    )


def load_split_state():
    req = [
        state_path("train_split.csv"),
        state_path("val_split.csv"),
        state_path("test_split.csv"),
        state_path("meta.json"),
    ]
    if not all(x.exists() for x in req):
        return None
    train_df = pd.read_csv(req[0])
    val_df = pd.read_csv(req[1])
    test_df = pd.read_csv(req[2])
    meta = json.loads(req[3].read_text(encoding="utf-8"))
    return (
        train_df,
        val_df,
        test_df,
        meta["label_to_idx"],
        {int(k): v for k, v in meta["idx_to_label"].items()},
        int(meta["num_classes"]),
    )


def maybe_download_dataset() -> None:
    os.makedirs(download_path, exist_ok=True)
    script_data = SCRIPT_DIR / "data"
    workspace_data = WORKSPACE_DIR / "data"
    candidates = [
        Path(download_path),
        Path(download_path) / "Vegetable Images",
        Path(download_path) / "misrakahmed_vegetable-image-dataset",
        Path(download_path) / "misrakahmed_vegetable-image-dataset" / "Vegetable Images",
        script_data,
        script_data / "Vegetable Images",
        workspace_data,
        workspace_data / "Vegetable Images",
    ]
    if any(p.exists() for p in candidates):
        return

    try:
        from kagglehub import dataset_download
    except Exception as ex:
        raise RuntimeError(
            "Dataset not found in expected path and kagglehub is unavailable. "
            "Install kagglehub or place dataset files under ./data/image"
        ) from ex

    print(f"Downloading {dataset_name}")
    dataset_download(dataset_name, output_dir=download_path)
    print("Download complete")


def find_dataset_root(candidates, known_splits, image_ext):
    for root in candidates:
        if not root.exists():
            continue
        class_dirs = [p for p in root.iterdir() if p.is_dir()]
        if not class_dirs:
            continue

        split_children = [p for p in class_dirs if p.name.lower() in known_splits]
        if split_children:
            class_count = 0
            for split_dir in split_children:
                class_count += len([p for p in split_dir.iterdir() if p.is_dir()])
            if class_count >= 5:
                return root

        valid = 0
        for cdir in class_dirs:
            if any(f.suffix.lower() in image_ext for f in cdir.rglob("*")):
                valid += 1
        if valid >= 5:
            return root
    return None


def _build_dataset_root_candidates() -> list[Path]:
    candidates = [
        Path(download_path),
        Path(download_path) / "Vegetable Images",
        Path(download_path) / "misrakahmed_vegetable-image-dataset",
        Path(download_path) / "misrakahmed_vegetable-image-dataset" / "Vegetable Images",
        SCRIPT_DIR / "data",
        SCRIPT_DIR / "data" / "Vegetable Images",
        WORKSPACE_DIR / "data",
        WORKSPACE_DIR / "data" / "Vegetable Images",
    ]
    seen = set()
    unique = []
    for c in candidates:
        key = str(c.resolve()) if c.exists() else str(c)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique


def _resolve_existing_image_path(path_text: str, root_candidates: list[Path]) -> str:
    raw = str(path_text)
    src = Path(raw)

    trial_paths = []
    if src.is_absolute():
        trial_paths.append(src)
    else:
        trial_paths.extend([
            (WORKSPACE_DIR / src),
            (SCRIPT_DIR / src),
            (Path.cwd() / src),
        ])

    parts_lower = [p.lower() for p in src.parts]
    if "vegetable images" in parts_lower:
        idx = parts_lower.index("vegetable images")
        suffix = Path(*src.parts[idx + 1 :]) if idx + 1 < len(src.parts) else Path()
        for root in root_candidates:
            if root.name.lower() == "vegetable images":
                trial_paths.append(root / suffix)
            else:
                trial_paths.append(root / "Vegetable Images" / suffix)

    seen = set()
    for p in trial_paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            try:
                return str(p.resolve())
            except Exception:
                return str(p)

    return raw


def _normalize_frame_filepaths(frame: pd.DataFrame, root_candidates: list[Path]) -> tuple[pd.DataFrame, int, int]:
    if frame.empty or "filepath" not in frame.columns:
        return frame, 0, 0

    fixed = frame.copy()
    before = fixed["filepath"].astype(str)
    fixed["filepath"] = before.map(lambda p: _resolve_existing_image_path(p, root_candidates))
    repaired = int((before != fixed["filepath"]).sum())

    exists_mask = fixed["filepath"].map(lambda p: Path(str(p)).exists())
    dropped = int((~exists_mask).sum())
    if dropped > 0:
        fixed = fixed[exists_mask].reset_index(drop=True)
    return fixed, repaired, dropped


def is_valid_image(path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


warnings.filterwarnings(
    "ignore",
    message=r".*isinstance\(treespec, LeafSpec\) is deprecated.*",
    category=DeprecationWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*train_dataloader.*does not have many workers.*",
    category=PossibleUserWarning,
)

in_ipykernel = "ipykernel" in sys.modules
SAFE_NUM_WORKERS = NUM_WORKERS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = None


def prepare_runtime_and_data() -> bool:
    global in_ipykernel, SAFE_NUM_WORKERS, DEVICE
    global train_df, val_df, test_df, label_to_idx, idx_to_label, NUM_CLASSES

    for p in [Path(download_path), LOG_DIR, CHECKPOINT_DIR, RESULT_DIR, STATE_DIR]:
        p.mkdir(parents=True, exist_ok=True)

    for env_candidate in (Path(".env"), Path("BTL1/.env")):
        if not env_candidate.exists():
            continue
        for raw_line in env_candidate.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    in_ipykernel = "ipykernel" in sys.modules
    if platform.system() == "Darwin" and in_ipykernel:
        SAFE_NUM_WORKERS = int(os.getenv("NOTEBOOK_NUM_WORKERS", "0"))
    else:
        SAFE_NUM_WORKERS = NUM_WORKERS
    SAFE_NUM_WORKERS = max(0, min(SAFE_NUM_WORKERS, NUM_WORKERS))

    mps_available = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    if platform.system() == "Darwin" and mps_available:
        DEVICE = torch.device("mps")
    elif platform.system() == "Windows" and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif mps_available:
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    update_pipeline_step(
        "runtime_setup",
        "completed",
        fingerprint=GLOBAL_CONFIG_FINGERPRINT,
        artifacts={
            "log_dir": str(LOG_DIR),
            "checkpoint_dir": str(CHECKPOINT_DIR),
            "result_dir": str(RESULT_DIR),
            "state_manifest": str(PIPELINE_STATE_FILE),
        },
    )

    print(f"Device: {DEVICE}")
    print(
        "timm: {} | tensorboard: {} | wandb: {} | lightning: {} | litlogger: {}".format(
            HAS_TIMM, HAS_TENSORBOARD, HAS_WANDB, HAS_LIGHTNING, USE_LITLOGGER
        )
    )
    print(f"AMP: {USE_AMP and DEVICE.type == 'cuda'} | channels_last: {USE_CHANNELS_LAST and DEVICE.type == 'cuda'}")
    print(f"Log dir: {LOG_DIR}")
    print(f"Checkpoint dir: {CHECKPOINT_DIR} | resume: {RESUME_FROM_CHECKPOINT}")
    print(f"Result dir: {RESULT_DIR}")
    print(f"W&B project: {WANDB_PROJECT} | mode: {WANDB_MODE} | entity: {WANDB_ENTITY}")
    print(f"DataLoader workers requested: {NUM_WORKERS} | effective: {SAFE_NUM_WORKERS}")

    if args.dry_run:
        print("Dry run complete.")
        return False

    maybe_download_dataset()

    candidate_roots = _build_dataset_root_candidates()
    image_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    known_splits = {"train", "validation", "val", "test"}

    index_cache_file = state_path("indexed_records.csv")
    bad_files_cache = state_path("indexed_bad_files.txt")
    index_fingerprint = build_config_fingerprint({"step": "dataset_indexing"})

    if step_is_completed("dataset_indexing", fingerprint=index_fingerprint, required_files=[index_cache_file]):
        df = pd.read_csv(index_cache_file)
        bad_files = bad_files_cache.read_text(encoding="utf-8").splitlines() if bad_files_cache.exists() else []
        dataset_root = Path(df["filepath"].iloc[0]).parents[2] if not df.empty else Path(download_path)
    else:
        dataset_root = find_dataset_root(candidate_roots, known_splits, image_ext)
        if dataset_root is None:
            checked = "\n".join(str(p) for p in candidate_roots)
            raise FileNotFoundError(
                "Dataset root with class folders was not found. Checked:\n" + checked
            )

        records = []
        bad_files = []
        for file in dataset_root.rglob("*"):
            if not file.is_file() or file.suffix.lower() not in image_ext:
                continue

            parts_lower = [p.lower() for p in file.parts]
            split_name = None
            if "train" in parts_lower:
                split_name = "train"
            elif "validation" in parts_lower or "val" in parts_lower:
                split_name = "val"
            elif "test" in parts_lower:
                split_name = "test"

            if is_valid_image(file):
                label = file.parent.name
                records.append({"filepath": str(file), "label": label, "split": split_name})
            else:
                bad_files.append(str(file))

        df = pd.DataFrame(records)
        if df.empty:
            raise ValueError("No valid image files found")

        df.to_csv(index_cache_file, index=False)
        bad_files_cache.write_text("\n".join(bad_files), encoding="utf-8")
        update_pipeline_step(
            "dataset_indexing",
            "completed",
            fingerprint=index_fingerprint,
            artifacts={"index_file": str(index_cache_file), "bad_files_file": str(bad_files_cache)},
            detail={"n_records": int(len(df)), "n_bad_files": int(len(bad_files))},
        )

    df, repaired_paths, dropped_paths = _normalize_frame_filepaths(df, candidate_roots)
    if repaired_paths > 0 or dropped_paths > 0:
        if repaired_paths > 0:
            print(f"Repaired {repaired_paths:,} cached file path(s) to absolute existing paths.")
        if dropped_paths > 0:
            warnings.warn(f"Dropped {dropped_paths:,} missing image path(s) from dataset index.")
        if df.empty:
            raise RuntimeError("All indexed image paths are missing after normalization.")
        df.to_csv(index_cache_file, index=False)

    label_to_idx = {label: idx for idx, label in enumerate(sorted(df["label"].unique()))}
    idx_to_label = {v: k for k, v in label_to_idx.items()}
    df["target"] = df["label"].map(label_to_idx)
    NUM_CLASSES = len(label_to_idx)

    print(f"Dataset root: {dataset_root}")
    print(f"Total valid images: {len(df):,}")
    print(f"Dropped corrupted images: {len(bad_files):,}")
    print(f"Classes: {NUM_CLASSES}")

    split_files = [
        state_path("train_split.csv"),
        state_path("val_split.csv"),
        state_path("test_split.csv"),
        state_path("meta.json"),
    ]
    if step_is_completed("split_state", required_files=split_files):
        restored = load_split_state()
        if restored is None:
            raise RuntimeError("Split state marked complete but files are missing.")
        train_df, val_df, test_df, label_to_idx, idx_to_label, NUM_CLASSES = restored
        print("Loaded split state from checkpoint/state.")
    else:
        if "split" in df.columns and set(["train", "val", "test"]).issubset(set(df["split"].dropna().unique())):
            train_df = df[df["split"] == "train"].copy()
            val_df = df[df["split"] == "val"].copy()
            test_df = df[df["split"] == "test"].copy()
        else:
            train_df, temp_df = train_test_split(df, test_size=0.30, random_state=SEED, stratify=df["target"])
            val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=SEED, stratify=temp_df["target"])

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        save_split_state(train_df, val_df, test_df, label_to_idx, idx_to_label)

    train_df, rep_train, drop_train = _normalize_frame_filepaths(train_df, candidate_roots)
    val_df, rep_val, drop_val = _normalize_frame_filepaths(val_df, candidate_roots)
    test_df, rep_test, drop_test = _normalize_frame_filepaths(test_df, candidate_roots)
    total_repaired = rep_train + rep_val + rep_test
    total_dropped = drop_train + drop_val + drop_test
    if total_repaired > 0 or total_dropped > 0:
        if total_repaired > 0:
            print(f"Repaired {total_repaired:,} split file path(s) to absolute existing paths.")
        if total_dropped > 0:
            warnings.warn(f"Dropped {total_dropped:,} missing image path(s) from split state.")
        if train_df.empty or val_df.empty or test_df.empty:
            raise RuntimeError("One of train/val/test splits is empty after path normalization.")
        save_split_state(train_df, val_df, test_df, label_to_idx, idx_to_label)

    if not HAS_LIGHTNING:
        warnings.warn("Lightning is unavailable. Falling back to native PyTorch training loop.")

    return True


class ImageClassificationDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        with Image.open(row["filepath"]) as im:
            image = im.convert("RGB")
        image = self.transform(image)
        return image, int(row["target"])


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


base_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

aug_tfms = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def build_loaders_from_splits():
    global train_loader, val_loader, test_loader, SAFE_NUM_WORKERS

    train_ds = ImageClassificationDataset(train_df, transform=aug_tfms)
    val_ds = ImageClassificationDataset(val_df, transform=base_tfms)
    test_ds = ImageClassificationDataset(test_df, transform=base_tfms)

    effective_workers = SAFE_NUM_WORKERS
    if platform.system() != "Darwin" and DEVICE.type in {"mps", "cuda"} and effective_workers < 2:
        effective_workers = min(2, max(1, (os.cpu_count() or 2) - 1))

    def _is_worker_resource_error(ex: Exception) -> bool:
        msg = str(ex).lower()
        return (
            "winerror 1455" in msg
            or "paging file is too small" in msg
            or "nvperf_host.dll" in msg
            or "openblas" in msg
            or "memory allocation" in msg
            or "worker exited unexpectedly" in msg
            or "dataloader timed out" in msg
            or "timed out after" in msg
        )

    def _build_with_workers(num_workers: int):
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": DEVICE.type == "cuda",
            "worker_init_fn": seed_worker,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR
            loader_kwargs["timeout"] = int(os.getenv("DATALOADER_TIMEOUT_SEC", "120"))

        g_train = torch.Generator().manual_seed(SEED)
        g_val = torch.Generator().manual_seed(SEED + 1)
        g_test = torch.Generator().manual_seed(SEED + 2)

        t_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g_train, **loader_kwargs)
        v_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, generator=g_val, **loader_kwargs)
        te_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, generator=g_test, **loader_kwargs)
        return t_loader, v_loader, te_loader

    workers_to_try = []
    w = int(effective_workers)
    while True:
        workers_to_try.append(w)
        if w == 0:
            break
        if w <= 2:
            w -= 1
        else:
            w //= 2

    last_error = None
    for num_workers in workers_to_try:
        try:
            t_loader, v_loader, te_loader = _build_with_workers(num_workers)
            # Force worker process start early so failures happen here, not in epoch loop.
            if num_workers > 0:
                _ = next(iter(t_loader))

            train_loader, val_loader, test_loader = t_loader, v_loader, te_loader
            if num_workers != effective_workers:
                warnings.warn(
                    f"DataLoader worker count reduced from {effective_workers} to {num_workers} "
                    "due to Windows virtual-memory pressure (WinError 1455)."
                )
                SAFE_NUM_WORKERS = num_workers
            return
        except Exception as ex:
            last_error = ex
            if not _is_worker_resource_error(ex) or num_workers == 0:
                raise
            warnings.warn(f"DataLoader worker startup failed with {num_workers} workers: {ex}")
            gc.collect()

    if last_error is not None:
        raise last_error


def build_resnet18(n):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, n)
    return model


def build_efficientnet_b0(n):
    model = torchvision.models.efficientnet_b0(weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    return model


def build_efficientnet_b4(n):
    model = torchvision.models.efficientnet_b4(weights=torchvision.models.EfficientNet_B4_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n)
    return model


def build_convnext_v2_tiny(n):
    weights = torchvision.models.ConvNeXt_V2_Tiny_Weights.DEFAULT
    model = torchvision.models.convnext_v2_tiny(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, n)
    return model


def build_convnext_v2_base(n):
    weights = torchvision.models.ConvNeXt_V2_Base_Weights.DEFAULT
    model = torchvision.models.convnext_v2_base(weights=weights)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, n)
    return model


def build_vit_b16(n):
    model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, n)
    return model


def build_vit_l16(n):
    model = torchvision.models.vit_l_16(weights=torchvision.models.ViT_L_16_Weights.DEFAULT)
    model.heads.head = nn.Linear(model.heads.head.in_features, n)
    return model


def build_swin_t(n):
    model = torchvision.models.swin_t(weights=torchvision.models.Swin_T_Weights.DEFAULT)
    model.head = nn.Linear(model.head.in_features, n)
    return model


def build_deit_tiny(n):
    if not HAS_TIMM:
        raise RuntimeError("timm is required for DeiT-Tiny")
    return timm.create_model("deit_tiny_patch16_224", pretrained=True, num_classes=n)


def build_mobilevit_xs(n):
    if not HAS_TIMM:
        raise RuntimeError("timm is required for MobileViT-XS")
    return timm.create_model("mobilevit_xs", pretrained=True, num_classes=n)


class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_class_weights(frame):
    counts = frame["target"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(float)
    inv = 1.0 / np.clip(counts.values, 1.0, None)
    weights = torch.tensor(inv, dtype=torch.float32)
    return weights / weights.mean()


def ensure_recovery_context():
    global train_df, val_df, test_df, label_to_idx, idx_to_label, NUM_CLASSES, class_weights
    if not all(x in globals() for x in ["train_df", "val_df", "test_df", "label_to_idx", "idx_to_label", "NUM_CLASSES"]):
        restored = load_split_state()
        if restored is None:
            raise RuntimeError("Split state not found. Restore checkpoint/image/.../state files.")
        train_df, val_df, test_df, label_to_idx, idx_to_label, NUM_CLASSES = restored
    class_weights = build_class_weights(train_df)
    if not all(x in globals() for x in ["train_loader", "val_loader", "test_loader"]):
        build_loaders_from_splits()


def checkpoint_path(model_name, finetune_mode):
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return CHECKPOINT_DIR / f"{safe}_{finetune_mode}_latest.pt"


def train_model_native(model, model_name, finetune_mode, run_fingerprint):
    ensure_recovery_context()

    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{model_name}_{finetune_mode}")
    run_dir = LOG_DIR / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_log = run_dir / "train_eval_log.csv"

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    initial_stage = "full" if finetune_mode == "scratch" else "head"
    set_finetune_stage(model, model_name, initial_stage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    history = []
    start_epoch = 1

    ckpt_file = checkpoint_path(model_name, finetune_mode)
    if RESUME_FROM_CHECKPOINT and ckpt_file.exists():
        state = torch.load(ckpt_file, map_location=DEVICE)
        if state.get("run_fingerprint") == run_fingerprint and state.get("finetune_mode") == finetune_mode:
            model.load_state_dict(state["model_state"])
            if state.get("optimizer_state") is not None:
                optimizer.load_state_dict(state["optimizer_state"])
            if state.get("scheduler_state") is not None:
                try:
                    scheduler.load_state_dict(state["scheduler_state"])
                except Exception:
                    pass
            if state.get("scaler_state") is not None:
                try:
                    scaler.load_state_dict(state["scaler_state"])
                except Exception:
                    pass

            best_val_loss = float(state.get("best_val_loss", best_val_loss))
            best_state = state.get("best_state")
            no_improve = int(state.get("no_improve", 0))
            history = list(state.get("history", []))
            start_epoch = int(state.get("epoch", 0)) + 1

    if start_epoch > EPOCHS:
        if best_state is not None:
            model.load_state_dict(best_state)
        hist_df = pd.DataFrame(history)
        if not hist_df.empty:
            hist_df.to_csv(csv_log, index=False)
        return model, hist_df, {
            "run_dir": str(run_dir),
            "csv_log": str(csv_log),
            "tensorboard_dir": None,
            "wandb_enabled": False,
            "wandb_mode": None,
            "event_log": None,
            "wandb_step": None,
            "lightning_checkpoint": None,
        }

    for epoch in range(start_epoch, EPOCHS + 1):
        if finetune_mode == "two_stage" and epoch == FREEZE_EPOCHS + 1:
            set_finetune_stage(model, model_name, "full")
            for param_group in optimizer.param_groups:
                param_group["lr"] = 1e-4
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS - FREEZE_EPOCHS))
            no_improve = 0
            best_val_loss = float("inf")

        model.train()
        run_loss, run_correct, run_total = 0.0, 0, 0
        train_iter = tqdm(train_loader, desc=f"{model_name} [{finetune_mode}] Epoch {epoch}/{EPOCHS}", leave=False)
        for xb, yb in train_iter:
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=USE_AMP and DEVICE.type == "cuda"):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            run_loss += loss.item() * xb.size(0)
            run_correct += (logits.argmax(dim=1) == yb).sum().item()
            run_total += xb.size(0)

        scheduler.step()

        train_loss = run_loss / max(1, run_total)
        train_acc = run_correct / max(1, run_total)
        val_metrics = evaluate_with_metrics(model, val_loader, criterion, progress_desc=f"{model_name} Val [{finetune_mode}] {epoch}/{EPOCHS}")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
            }
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if SAVE_CHECKPOINT_EVERY_EPOCH:
            torch.save(
                {
                    "epoch": epoch,
                    "epoch_seal": "validation_complete",
                    "finetune_mode": finetune_mode,
                    "run_fingerprint": run_fingerprint,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_state": best_state,
                    "no_improve": no_improve,
                    "history": history,
                    "logger_run_dir": str(run_dir),
                    "lightning_checkpoint": None,
                },
                ckpt_file,
            )

        if no_improve >= 2:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    hist_df = pd.DataFrame(history)
    if not hist_df.empty:
        hist_df.to_csv(csv_log, index=False)

    return model, hist_df, {
        "run_dir": str(run_dir),
        "csv_log": str(csv_log),
        "tensorboard_dir": None,
        "wandb_enabled": False,
        "wandb_mode": None,
        "event_log": None,
        "wandb_step": None,
        "lightning_checkpoint": None,
    }


def evaluate_with_metrics(model, loader, criterion, progress_desc="Evaluating"):
    model = model.to(DEVICE)
    model.eval()
    loss_sum = 0.0
    y_true, y_pred, y_prob = [], [], []
    use_amp = USE_AMP and DEVICE.type in {"cuda", "mps"}

    with torch.inference_mode():
        iterator = tqdm(loader, desc=progress_desc, leave=False)
        for xb, yb in iterator:
            xb = xb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))
            if USE_CHANNELS_LAST and DEVICE.type == "cuda":
                xb = xb.contiguous(memory_format=torch.channels_last)
            yb = yb.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp and DEVICE.type == "cuda"):
                logits = model(xb)
                probs = torch.softmax(logits, dim=1)
                loss = criterion(logits, yb)

            loss_sum += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            y_true.extend(yb.detach().cpu().numpy().tolist())
            y_pred.extend(preds.detach().cpu().numpy().tolist())
            y_prob.extend(probs.detach().cpu().numpy().tolist())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    metrics = {
        "loss": loss_sum / len(loader.dataset),
        "accuracy": float((y_true == y_pred).mean()),
        "top5_accuracy": float(top_k_accuracy_score(y_true, y_prob, k=min(5, NUM_CLASSES), labels=np.arange(NUM_CLASSES))),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    try:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"))
    except Exception:
        metrics["roc_auc_ovr_macro"] = np.nan

    return metrics


def _unfreeze_module(module):
    if module is None:
        return False
    has_params = False
    for p in module.parameters():
        p.requires_grad = True
        has_params = True
    return has_params


def _unfreeze_classifier_like(model):
    candidates = []
    if hasattr(model, "classifier"):
        candidates.append(getattr(model, "classifier"))
    if hasattr(model, "head"):
        candidates.append(getattr(model, "head"))
    if hasattr(model, "heads"):
        candidates.append(getattr(model, "heads"))
    if hasattr(model, "get_classifier"):
        try:
            candidates.append(model.get_classifier())
        except Exception:
            pass

    seen = set()
    for module in candidates:
        if module is None:
            continue
        key = id(module)
        if key in seen:
            continue
        seen.add(key)
        if isinstance(module, nn.Module) and _unfreeze_module(module):
            return True

    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear) and _unfreeze_module(module):
            return True
    return False


def set_finetune_stage(model, model_name, stage):
    if model_name == "Basic CNN":
        for p in model.parameters():
            p.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = False

    if model_name in ["ResNet18"]:
        _unfreeze_module(getattr(model, "fc", None))
    elif model_name in ["EfficientNet-B0", "EfficientNet-B4"]:
        _unfreeze_module(getattr(model, "classifier", None))
    elif model_name in ["ConvNeXt-V2 Tiny", "ConvNeXt-V2 Base"]:
        _unfreeze_classifier_like(model)
    elif model_name in ["ViT-B/16", "ViT-L/16"]:
        _unfreeze_module(getattr(model, "heads", None))
    elif model_name in ["Swin-T", "DeiT-Tiny", "MobileViT-XS"]:
        _unfreeze_classifier_like(model)

    if stage == "full":
        for p in model.parameters():
            p.requires_grad = True


def mode_benchmark_state_file(mode):
    return state_path(f"benchmark_state_{mode}.pkl")


def _persist_mode_state(mode_state_file, results, mode_histories, mode_predictions, logging_rows):
    pd.to_pickle(
        {
            "results": results,
            "histories": mode_histories,
            "predictions": mode_predictions,
            "logging_rows": logging_rows,
        },
        mode_state_file,
    )


def _build_convnext_tiny_compatible(n):
    if not HAS_TIMM:
        raise RuntimeError("timm is required for ConvNeXt-V2 Tiny. Install timm and rerun.")
    model = timm.create_model("convnextv2_tiny.fcmae_ft_in22k_in1k", pretrained=True)
    model.reset_classifier(num_classes=n)
    return model


def _build_convnext_base_compatible(n):
    if not HAS_TIMM:
        raise RuntimeError("timm is required for ConvNeXt-V2 Base. Install timm and rerun.")
    model = timm.create_model("convnextv2_base.fcmae_ft_in22k_in1k", pretrained=True)
    model.reset_classifier(num_classes=n)
    return model


def _is_mps_oom_error(ex):
    msg = str(ex).lower()
    return DEVICE.type == "mps" and (
        "out of memory" in msg or "mps backend out of memory" in msg or ("mps" in msg and "memory" in msg)
    )


def _downshift_batch_size_for_mps(model_name):
    global BATCH_SIZE
    if DEVICE.type != "mps" or BATCH_SIZE <= 1:
        return False
    next_batch = max(1, BATCH_SIZE // 2)
    if next_batch == BATCH_SIZE:
        next_batch = BATCH_SIZE - 1
    if next_batch < 1:
        return False

    print(f"  -> MPS OOM detected for {model_name}; downshifting batch size {BATCH_SIZE} -> {next_batch} and retrying")
    BATCH_SIZE = next_batch
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    build_loaders_from_splits()
    return True


def get_model_zoo():
    zoo = [
        ("Basic CNN", "CNN", lambda n: BasicCNN(n)),
        ("ResNet18", "CNN", lambda n: build_resnet18(n)),
        ("EfficientNet-B0", "CNN", lambda n: build_efficientnet_b0(n)),
        ("EfficientNet-B4", "CNN", lambda n: build_efficientnet_b4(n)),
        ("ConvNeXt-V2 Tiny", "CNN", lambda n: _build_convnext_tiny_compatible(n)),
        ("ConvNeXt-V2 Base", "CNN", lambda n: _build_convnext_base_compatible(n)),
        ("ViT-B/16", "ViT", lambda n: build_vit_b16(n)),
        ("ViT-L/16", "ViT", lambda n: build_vit_l16(n)),
        ("Swin-T", "ViT", lambda n: build_swin_t(n)),
    ]
    if HAS_TIMM:
        zoo.extend(
            [
                ("DeiT-Tiny", "ViT", lambda n: build_deit_tiny(n)),
                ("MobileViT-XS", "ViT", lambda n: build_mobilevit_xs(n)),
            ]
        )
    if not RUN_FULL_MODEL_SET:
        zoo = [
            ("Basic CNN", "CNN", lambda n: BasicCNN(n)),
            ("ResNet18", "CNN", lambda n: build_resnet18(n)),
            ("EfficientNet-B0", "CNN", lambda n: build_efficientnet_b0(n)),
            ("ViT-B/16", "ViT", lambda n: build_vit_b16(n)),
        ]
    return zoo


if HAS_LIGHTNING:
    LightningCallbackBase = pl.Callback
    LightningModuleBase = pl.LightningModule
else:
    LightningCallbackBase = object
    LightningModuleBase = nn.Module


class HistoryCallback(LightningCallbackBase):
    def __init__(self):
        super().__init__()
        self.rows = []

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics

        def _f(key, default=np.nan):
            v = m.get(key)
            if v is None:
                return float(default)
            if isinstance(v, torch.Tensor):
                return float(v.detach().cpu().item())
            try:
                return float(v)
            except Exception:
                return float(default)

        self.rows.append(
            {
                "epoch": int(trainer.current_epoch + 1),
                "train_loss": _f("train/loss_epoch"),
                "train_acc": _f("train/acc_epoch"),
                "val_loss": _f("val/loss"),
                "val_acc": _f("val/acc"),
                "val_f1_macro": _f("val/f1_macro"),
            }
        )


class WandbMiniStepSyncCallback(LightningCallbackBase):
    def __init__(self, run_dir, run_name):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.run_name = run_name

    def on_validation_epoch_end(self, trainer, pl_module):
        if not WANDB_AUTO_SYNC_MINISTEP:
            return
        every_n = max(1, WANDB_SYNC_EVERY_N_MINISTEPS)
        epoch = int(trainer.current_epoch + 1)
        if epoch % every_n != 0:
            return
        wandb_dir = self.run_dir / "wandb"
        if not wandb_dir.exists():
            return
        cmd = [sys.executable, "-m", "wandb", "sync", "--sync-all", str(wandb_dir)]
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=WANDB_SYNC_TIMEOUT)
            if res.returncode == 0 and WANDB_RECOVER_ONLINE and WANDB_MODE == "offline":
                print(f"W&B mode switched back to online after sync at epoch {epoch} for {self.run_name}")
        except Exception as ex:
            print(f"W&B sync skipped for {self.run_name}: {ex}")


class LightningImageClassifier(LightningModuleBase):
    def __init__(self, model, model_name, finetune_mode, class_weights, lr=3e-4, weight_decay=1e-4):
        super().__init__()
        self.model = model
        self.model_name = model_name
        self.finetune_mode = finetune_mode
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.val_preds = []
        self.val_targets = []

        initial_stage = "full" if finetune_mode == "scratch" else "head"
        set_finetune_stage(self.model, self.model_name, initial_stage)

    def on_train_epoch_start(self):
        epoch = int(self.current_epoch + 1)
        if self.finetune_mode == "two_stage" and epoch == FREEZE_EPOCHS + 1:
            set_finetune_stage(self.model, self.model_name, "full")
            for pg in self.trainer.optimizers[0].param_groups:
                pg["lr"] = 1e-4

    def forward(self, x):
        return self.model(x)

    def _prep_batch(self, batch):
        xb, yb = batch
        if USE_CHANNELS_LAST and self.device.type == "cuda":
            xb = xb.contiguous(memory_format=torch.channels_last)
        return xb, yb

    def training_step(self, batch, batch_idx):
        xb, yb = self._prep_batch(batch)
        logits = self(xb)
        loss = self.criterion(logits, yb)
        acc = (logits.argmax(dim=1) == yb).float().mean()
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=xb.size(0))
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True, batch_size=xb.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        xb, yb = self._prep_batch(batch)
        logits = self(xb)
        loss = self.criterion(logits, yb)
        preds = logits.argmax(dim=1)
        acc = (preds == yb).float().mean()
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=xb.size(0))
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=xb.size(0))
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(yb.detach().cpu())

    def on_validation_epoch_end(self):
        if not self.val_preds:
            return
        y_pred = torch.cat(self.val_preds).numpy()
        y_true = torch.cat(self.val_targets).numpy()
        f1_macro = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2]
        self.log("val/f1_macro", float(f1_macro), on_step=False, on_epoch=True, prog_bar=True)
        self.val_preds.clear()
        self.val_targets.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, EPOCHS))
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }


def lightning_checkpoint_path(model_name, finetune_mode):
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", model_name)
    return CHECKPOINT_DIR / f"{safe}_{finetune_mode}_lightning_last.ckpt"


def train_model_lightning(model, model_name, finetune_mode, run_fingerprint):
    if not HAS_LIGHTNING:
        raise RuntimeError("Lightning is unavailable. Use native training loop instead.")

    ensure_recovery_context()

    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{model_name}_{finetune_mode}")
    run_key = re.sub(r"[^a-z0-9-]+", "-", f"{finetune_mode}-{model_name}".lower()).strip("-")
    run_id = run_key
    run_dir = LOG_DIR / safe
    run_dir.mkdir(parents=True, exist_ok=True)

    lit_ckpt = lightning_checkpoint_path(model_name, finetune_mode)
    legacy_ckpt = checkpoint_path(model_name, finetune_mode)
    csv_logger = PLCSVLogger(save_dir=str(run_dir), name="pl_csv")
    loggers = [csv_logger]

    if USE_LITLOGGER and HAS_LITLOGGER and not in_ipykernel:
        try:
            loggers.append(PLLitLogger())
        except Exception as ex:
            print(f"LitLogger disabled for {safe}: {ex}")

    wandb_logger = None
    used_wandb_mode = None
    if USE_WANDB and HAS_WANDB:
        requested_mode = WANDB_MODE
        try_modes = [requested_mode]
        if requested_mode != "online":
            try_modes.append("online")
        if "offline" not in try_modes:
            try_modes.append("offline")

        settings = wandb.Settings(init_timeout=WANDB_INIT_TIMEOUT)
        for mode in try_modes:
            try:
                wandb_logger = PLWandbLogger(
                    project=WANDB_PROJECT,
                    entity=WANDB_ENTITY,
                    name=run_key,
                    save_dir=str(run_dir),
                    mode=mode,
                    log_model=False,
                    id=run_id,
                    resume="allow",
                    settings=settings,
                )
                wandb_logger.log_hyperparams(
                    {
                        "model": model_name,
                        "finetune_mode": finetune_mode,
                        "epochs": EPOCHS,
                        "freeze_epochs": FREEZE_EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "img_size": IMG_SIZE,
                        "run_fingerprint": run_fingerprint,
                        "wandb_mode_used": mode,
                    }
                )
                used_wandb_mode = mode
                loggers.append(wandb_logger)
                if mode != requested_mode:
                    print(f"W&B fallback activated for {safe}: {requested_mode} -> {mode}")
                break
            except Exception as ex:
                wandb_logger = None
                print(f"W&B logger init failed for {safe} in mode={mode}: {ex}")

    history_cb = HistoryCallback()
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CHECKPOINT_DIR),
        filename=f"{re.sub(r'[^a-zA-Z0-9_.-]+', '_', model_name)}_{finetune_mode}_lightning_{{epoch:02d}}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        every_n_epochs=1,
    )

    callbacks = [checkpoint_cb, history_cb, TQDMProgressBar(refresh_rate=20)]
    if WANDB_AUTO_SYNC_MINISTEP:
        callbacks.append(WandbMiniStepSyncCallback(run_dir=run_dir, run_name=safe))

    lit_module = LightningImageClassifier(
        model=model,
        model_name=model_name,
        finetune_mode=finetune_mode,
        class_weights=class_weights.to(DEVICE),
    )

    accelerator = "gpu" if DEVICE.type == "cuda" else ("mps" if DEVICE.type == "mps" else "cpu")
    precision = "16-mixed" if (USE_AMP and DEVICE.type == "cuda") else "32-true"

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=1,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=False,
        precision=precision,
    )

    ckpt_path = str(lit_ckpt) if (RESUME_FROM_CHECKPOINT and lit_ckpt.exists()) else None
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    trainer.save_checkpoint(str(lit_ckpt))

    trained_model = lit_module.model.to(DEVICE)
    trained_model.eval()

    hist = pd.DataFrame(history_cb.rows)
    if not hist.empty:
        hist = hist.sort_values("epoch").drop_duplicates(subset=["epoch"], keep="last").reset_index(drop=True)

    torch.save(
        {
            "epoch": int(hist["epoch"].max()) if not hist.empty else EPOCHS,
            "epoch_seal": "validation_complete",
            "finetune_mode": finetune_mode,
            "run_fingerprint": run_fingerprint,
            "model_state": trained_model.state_dict(),
            "history": history_cb.rows,
            "lightning_checkpoint": str(lit_ckpt),
            "logger_run_dir": str(run_dir),
            "wandb_mode": used_wandb_mode,
        },
        legacy_ckpt,
    )

    if wandb_logger is not None:
        try:
            wandb_logger.experiment.finish()
        except Exception:
            pass

    return trained_model, hist, {
        "run_dir": str(run_dir),
        "csv_log": str(Path(csv_logger.log_dir) / "metrics.csv"),
        "tensorboard_dir": None,
        "wandb_enabled": bool(wandb_logger is not None),
        "wandb_mode": used_wandb_mode,
        "event_log": None,
        "wandb_step": None,
        "lightning_checkpoint": str(lit_ckpt),
    }


def run_benchmark() -> None:
    all_results = []
    all_logging_rows = []
    histories = {}
    predictions = {}

    for finetune_mode in FINETUNE_MODES:
        print(f"\n===== Finetune mode: {finetune_mode} =====")
        mode_fingerprint = build_config_fingerprint({"step": "benchmark", "finetune_mode": finetune_mode})
        mode_state_file = mode_benchmark_state_file(finetune_mode)

        if mode_state_file.exists():
            state = pd.read_pickle(mode_state_file)
            results = state.get("results", [])
            mode_histories = state.get("histories", {})
            mode_predictions = state.get("predictions", {})
            logging_rows = state.get("logging_rows", [])
        else:
            results, mode_histories, mode_predictions, logging_rows = [], {}, {}, []

        completed_keys = {(row.get("model"), row.get("finetune_mode"), row.get("run_fingerprint")) for row in results}
        if completed_keys:
            completed_models = sorted({x[0] for x in completed_keys if x[1] == finetune_mode})
            print(f"Resuming mode {finetune_mode}; completed models: {completed_models}")

        model_zoo = get_model_zoo()
        for model_name, family, builder in tqdm(model_zoo, desc=f"Training model zoo [{finetune_mode}]"):
            model_key = (model_name, finetune_mode, mode_fingerprint)
            if model_key in completed_keys:
                print(f"Skipping completed model: {model_name} [{finetune_mode}]")
                continue

            print(f"\nTraining {model_name} [{family}] with finetune_mode={finetune_mode}")
            try:
                model = builder(NUM_CLASSES)
            except Exception as ex:
                print(f"Skipped {model_name}: {ex}")
                continue

            try:
                while True:
                    try:
                        t0 = time.time()
                        print(f"  -> Starting fit for {model_name} [{finetune_mode}] | batch_size={BATCH_SIZE}")
                        if HAS_LIGHTNING:
                            model, hist, log_info = train_model_lightning(
                                model,
                                model_name,
                                finetune_mode=finetune_mode,
                                run_fingerprint=mode_fingerprint,
                            )
                        else:
                            model, hist, log_info = train_model_native(
                                model,
                                model_name,
                                finetune_mode=finetune_mode,
                                run_fingerprint=mode_fingerprint,
                            )
                        model = model.to(DEVICE)
                        fit_elapsed = time.time() - t0
                        print(f"  -> Fit complete in {fit_elapsed:.1f}s")

                        eval_t0 = time.time()
                        print(f"  -> Starting test evaluation for {model_name} [{finetune_mode}]")
                        criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
                        test_metrics = evaluate_with_metrics(
                            model,
                            test_loader,
                            criterion,
                            progress_desc=f"{model_name} Test [{finetune_mode}]",
                        )
                        eval_elapsed = time.time() - eval_t0
                        print(f"  -> Test evaluation complete in {eval_elapsed:.1f}s")
                        break
                    except RuntimeError as ex:
                        if not _is_mps_oom_error(ex):
                            raise
                        if not _downshift_batch_size_for_mps(model_name):
                            raise
                        model = builder(NUM_CLASSES)

                eval_epoch = int(hist["epoch"].max()) if not hist.empty else EPOCHS

                mode_histories[model_name] = hist
                mode_predictions[model_name] = test_metrics

                logging_rows.append(
                    {
                        "model": model_name,
                        "family": family,
                        "finetune_mode": finetune_mode,
                        "run_fingerprint": mode_fingerprint,
                        "run_dir": log_info["run_dir"],
                        "csv_log": log_info["csv_log"],
                        "tensorboard_dir": log_info["tensorboard_dir"],
                        "wandb_enabled": log_info["wandb_enabled"],
                        "event_log": log_info.get("event_log"),
                        "wandb_step": log_info.get("wandb_step"),
                        "lightning_checkpoint": log_info.get("lightning_checkpoint"),
                    }
                )

                results.append(
                    {
                        "model": model_name,
                        "family": family,
                        "finetune_mode": finetune_mode,
                        "run_fingerprint": mode_fingerprint,
                        "checkpoint_file": str(checkpoint_path(model_name, finetune_mode)),
                        "epochs_ran": eval_epoch,
                        "train_time_sec": fit_elapsed,
                        "val_acc_best": float(hist["val_acc"].max()) if (not hist.empty and "val_acc" in hist.columns) else np.nan,
                        "test_loss": float(test_metrics["loss"]),
                        "test_acc": float(test_metrics["accuracy"]),
                        "top5_acc": float(test_metrics["top5_accuracy"]),
                        "precision_macro": float(test_metrics["precision_macro"]),
                        "recall_macro": float(test_metrics["recall_macro"]),
                        "f1_macro": float(test_metrics["f1_macro"]),
                        "precision_weighted": float(test_metrics["precision_weighted"]),
                        "recall_weighted": float(test_metrics["recall_weighted"]),
                        "f1_weighted": float(test_metrics["f1_weighted"]),
                        "balanced_acc": float(test_metrics["balanced_accuracy"]),
                        "roc_auc_ovr_macro": float(test_metrics["roc_auc_ovr_macro"])
                        if pd.notna(test_metrics["roc_auc_ovr_macro"])
                        else np.nan,
                    }
                )

                _persist_mode_state(mode_state_file, results, mode_histories, mode_predictions, logging_rows)
                print(f"  -> Mode state saved: {mode_state_file}")

            except KeyboardInterrupt:
                _persist_mode_state(mode_state_file, results, mode_histories, mode_predictions, logging_rows)
                print(f"Interrupted. Benchmark state saved to: {mode_state_file}")
                raise
            except Exception as ex:
                _persist_mode_state(mode_state_file, results, mode_histories, mode_predictions, logging_rows)
                print(f"Failed {model_name} [{finetune_mode}] after training/build: {ex}")
                continue

        all_results.extend(results)
        all_logging_rows.extend(logging_rows)
        histories.update({f"{k}__{finetune_mode}": v for k, v in mode_histories.items()})
        predictions.update({f"{k}__{finetune_mode}": v for k, v in mode_predictions.items()})
        update_pipeline_step(
            f"benchmark_{finetune_mode}",
            "completed",
            fingerprint=mode_fingerprint,
            artifacts={"state_file": str(mode_state_file)},
        )

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        results_df = results_df.sort_values(["test_acc", "f1_macro"], ascending=False).reset_index(drop=True)
    logging_df = pd.DataFrame(all_logging_rows)

    all_models_path = RESULT_DIR / "all_models.csv"
    logging_path = RESULT_DIR / "all_logging.csv"
    if not results_df.empty:
        cols = [
            "model",
            "family",
            "finetune_mode",
            "run_fingerprint",
            "checkpoint_file",
            "epochs_ran",
            "train_time_sec",
            "val_acc_best",
            "test_loss",
            "test_acc",
            "top5_acc",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "precision_weighted",
            "recall_weighted",
            "f1_weighted",
            "balanced_acc",
            "roc_auc_ovr_macro",
        ]
        for c in cols:
            if c not in results_df.columns:
                results_df[c] = np.nan
        results_df = results_df[cols]
        results_df.to_csv(all_models_path, index=False)
        print(f"Saved consolidated results: {all_models_path}")

    if not logging_df.empty:
        logging_df.to_csv(logging_path, index=False)
        print(f"Saved consolidated logging map: {logging_path}")

    if not results_df.empty:
        print(results_df[["model", "finetune_mode", "test_acc", "f1_macro", "train_time_sec"]].head(30).to_string(index=False))


def main() -> None:
    global class_weights

    if not prepare_runtime_and_data():
        return

    build_loaders_from_splits()
    update_pipeline_step(
        "dataloader_ready",
        "completed",
        fingerprint=build_config_fingerprint({"step": "dataloader_ready"}),
        detail={
            "train_batches": int(len(train_loader)),
            "val_batches": int(len(val_loader)),
            "test_batches": int(len(test_loader)),
        },
    )
    class_weights = build_class_weights(train_df)
    run_benchmark()


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
