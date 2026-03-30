from __future__ import annotations

"""Shared coursework utilities for the recurrent constitutive-model workflow.

The file keeps the original skeleton naming where possible, then layers on the
additional machinery needed for the coursework: deterministic splits, train-only
normalisation, recurrent-core variants, checkpointing, hidden-size studies,
inference, and report plots.
"""

import copy
import json
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from statsmodels.tsa.stattools import acf, adfuller, pacf

    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    acf = None
    adfuller = None
    pacf = None


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT_DIR / "Coursework3" / "viscodata_3mat.mat"
F_FIELD = "epsi_tol"
SIG_FIELD = "sigma_tol"

DEFAULT_ARTIFACT_ROOT = ROOT_DIR / "artifacts"
DEFAULT_SPLIT_SEED = 20260328
DEFAULT_TRAIN_RATIO = 0.70
DEFAULT_VAL_RATIO = 0.15
DEFAULT_MPLCONFIGDIR = DEFAULT_ARTIFACT_ROOT / "mplconfig"
DEFAULT_CACHE_ROOT = DEFAULT_ARTIFACT_ROOT / ".cache"

DEFAULT_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Ntotal = 400
train_size = int(round(Ntotal * DEFAULT_TRAIN_RATIO))
val_size = int(round(Ntotal * DEFAULT_VAL_RATIO))
test_start = train_size + val_size
N_test = Ntotal - test_start

USE_CUDA = torch.cuda.is_available()
USE_MPS = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_matplotlib_cache(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> Path:
    mplconfigdir = ensure_directory(Path(artifact_root) / "mplconfig")
    cache_root = ensure_directory(Path(artifact_root) / ".cache")
    os.environ["MPLCONFIGDIR"] = str(mplconfigdir)
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    return mplconfigdir


def ensure_artifact_tree(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Path]:
    artifact_root = ensure_directory(artifact_root)
    directories = {
        "root": artifact_root,
        "environment": ensure_directory(artifact_root / "environment"),
        "eda": ensure_directory(artifact_root / "eda"),
        "splits": ensure_directory(artifact_root / "splits"),
        "normalization": ensure_directory(artifact_root / "normalization"),
        "checkpoints": ensure_directory(artifact_root / "checkpoints"),
        "logs": ensure_directory(artifact_root / "logs"),
        "predictions": ensure_directory(artifact_root / "predictions"),
        "figures": ensure_directory(artifact_root / "figures"),
        "optuna": ensure_directory(artifact_root / "optuna"),
        "final": ensure_directory(artifact_root / "final"),
        "reports": ensure_directory(artifact_root / "reports"),
    }
    return directories


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_mps: bool = True) -> torch.device:
    device_override = os.environ.get("COURSEWORK_DEVICE", "").strip().lower()
    if device_override:
        if device_override == "auto":
            device_override = ""
        elif device_override == "cpu":
            return torch.device("cpu")
        elif device_override == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            raise RuntimeError("COURSEWORK_DEVICE=cuda was requested, but CUDA is not available.")
        elif device_override == "mps":
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            raise RuntimeError("COURSEWORK_DEVICE=mps was requested, but MPS is not available.")
        else:
            raise ValueError(f"Unsupported COURSEWORK_DEVICE value: {device_override}")

    if prefer_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_numpy(array: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def tensor_to_device(array: np.ndarray | torch.Tensor, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(array, torch.Tensor):
        return array.to(device=device, dtype=dtype)
    return torch.as_tensor(array, device=device, dtype=dtype)


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


class DenseNet(nn.Module):
    def __init__(self, layers: list[int], nonlinearity: type[nn.Module]):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))
            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class MatReader(object):
    """Load MATLAB `.mat` files from either the classic or HDF5-backed formats."""

    def __init__(self, file_path: str | Path, to_torch: bool = True, to_cuda: bool = False, to_float: bool = True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self.file_path = str(file_path)
        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self) -> None:
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path: str | Path) -> None:
        self.file_path = str(file_path)
        self._load_file()

    def read_field(self, field: str) -> np.ndarray | torch.Tensor:
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda: bool) -> None:
        self.to_cuda = to_cuda

    def set_torch(self, to_torch: bool) -> None:
        self.to_torch = to_torch

    def set_float(self, to_float: bool) -> None:
        self.to_float = to_float


class MinMaxNormalizer:
    """Fit and reuse the train-only min/max scaling used throughout the workflow."""

    def __init__(self, feature_min: float = -1.0, feature_max: float = 1.0):
        self.feature_min = feature_min
        self.feature_max = feature_max
        self.data_min: float | None = None
        self.data_max: float | None = None

    def fit(self, x: np.ndarray | torch.Tensor) -> "MinMaxNormalizer":
        values = to_numpy(x)
        self.data_min = float(values.min())
        self.data_max = float(values.max())
        return self

    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert self.data_min is not None and self.data_max is not None
        denom = self.data_max - self.data_min
        if abs(denom) < 1e-12:
            denom = 1.0

        scale = (self.feature_max - self.feature_min) / denom
        if isinstance(x, torch.Tensor):
            data_min = torch.tensor(self.data_min, device=x.device, dtype=x.dtype)
            scale_tensor = torch.tensor(scale, device=x.device, dtype=x.dtype)
            feature_min = torch.tensor(self.feature_min, device=x.device, dtype=x.dtype)
            return (x - data_min) * scale_tensor + feature_min
        return (np.asarray(x) - self.data_min) * scale + self.feature_min

    def inverse_transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert self.data_min is not None and self.data_max is not None
        scale = self.feature_max - self.feature_min
        if abs(scale) < 1e-12:
            scale = 1.0
        denom = self.data_max - self.data_min
        if isinstance(x, torch.Tensor):
            feature_min = torch.tensor(self.feature_min, device=x.device, dtype=x.dtype)
            scale_tensor = torch.tensor(scale, device=x.device, dtype=x.dtype)
            denom_tensor = torch.tensor(denom, device=x.device, dtype=x.dtype)
            data_min = torch.tensor(self.data_min, device=x.device, dtype=x.dtype)
            return ((x - feature_min) / scale_tensor) * denom_tensor + data_min
        return ((np.asarray(x) - self.feature_min) / scale) * denom + self.data_min

    def to_dict(self) -> dict[str, float]:
        assert self.data_min is not None and self.data_max is not None
        return {
            "feature_min": self.feature_min,
            "feature_max": self.feature_max,
            "data_min": self.data_min,
            "data_max": self.data_max,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "MinMaxNormalizer":
        normalizer = cls(feature_min=payload["feature_min"], feature_max=payload["feature_max"])
        normalizer.data_min = payload["data_min"]
        normalizer.data_max = payload["data_max"]
        return normalizer


class ZScoreNormalizer:
    """Simple z-score normaliser used only for EDA comparisons."""

    def __init__(self):
        self.mean: float | None = None
        self.std: float | None = None

    def fit(self, x: np.ndarray | torch.Tensor) -> "ZScoreNormalizer":
        values = to_numpy(x)
        self.mean = float(values.mean())
        self.std = float(values.std() + 1e-8)
        return self

    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert self.mean is not None and self.std is not None
        if isinstance(x, torch.Tensor):
            mean = torch.tensor(self.mean, device=x.device, dtype=x.dtype)
            std = torch.tensor(self.std, device=x.device, dtype=x.dtype)
            return (x - mean) / std
        return (np.asarray(x) - self.mean) / self.std


class RNO(nn.Module):
    """Recurrent constitutive operator with interchangeable hidden-state mechanisms."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        layer_input: list[int],
        layer_hidden: list[int],
        core_type: str = "rnn",
        paper_use_rate_in_stress: bool = False,
    ):
        super(RNO, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.core_type = core_type.lower()
        self.has_hidden_state = hidden_size > 0
        self.is_baseline_rno = self.core_type == "baseline_rno"
        self.is_paper_rno = self.core_type == "paper_rno"
        self.paper_use_rate_in_stress = bool(paper_use_rate_in_stress)

        self.layers = DenseNet(layer_input, nn.SELU)

        if (self.is_baseline_rno or self.is_paper_rno) and not self.has_hidden_state:
            self.hidden_layers = None
            recurrent_input_size = input_size + output_size + 1
        elif len(layer_hidden) >= 2:
            self.hidden_layers = DenseNet(layer_hidden, nn.SELU)
            recurrent_input_size = layer_hidden[-1]
        else:
            self.hidden_layers = nn.Identity()
            recurrent_input_size = input_size + output_size + 1

        if self.is_baseline_rno or self.is_paper_rno:
            self.recurrent_cell = None
        elif not self.has_hidden_state:
            self.recurrent_cell = None
        elif self.core_type == "rnn":
            self.recurrent_cell = nn.RNNCell(recurrent_input_size, hidden_size, nonlinearity="tanh")
        elif self.core_type == "gru":
            self.recurrent_cell = nn.GRUCell(recurrent_input_size, hidden_size)
        elif self.core_type == "lstm":
            self.recurrent_cell = nn.LSTMCell(recurrent_input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported core_type: {core_type}")

    def forward(
        self,
        input: torch.Tensor,
        output: torch.Tensor,
        hidden: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor | tuple[torch.Tensor, torch.Tensor]]:
        if self.is_baseline_rno:
            if not self.has_hidden_state:
                # h=0 baseline: only the previous strain and its discrete rate reach the readout.
                combined = torch.cat((output, (output - input) / dt), dim=1)
                x = self.layers(combined)
                output = x.squeeze(1)
                return output, hidden
            # Skeleton-style hidden update: explicit Euler step on the learned internal variables.
            hidden_prev = hidden
            hidden_input = torch.cat((output, hidden_prev), dim=1)
            hidden_update = self.hidden_layers(hidden_input)
            hidden_next = hidden_prev + dt * hidden_update
            combined = torch.cat((output, (output - input) / dt, hidden_prev), dim=1)
            x = self.layers(combined)
            output = x.squeeze(1)
            return output, hidden_next

        if self.is_paper_rno:
            rate = (input - output) / dt
            if self.has_hidden_state:
                # Paper-style latent state evolution is driven by the current strain state.
                hidden_prev = hidden
                hidden_input = torch.cat((input, hidden_prev), dim=1)
                hidden_update = self.hidden_layers(hidden_input)
                hidden_next = hidden_prev + dt * hidden_update
                hidden_for_output = hidden_next
            else:
                # h=0 keeps the paper-style model honest: no latent memory is carried in time.
                hidden_next = hidden
                hidden_for_output = input.new_zeros((input.shape[0], 0))

            combined_parts = [input]
            if self.paper_use_rate_in_stress:
                # The viscoelastic paper variant exposes rate only in the stress readout.
                combined_parts.append(rate)
            if self.has_hidden_state:
                combined_parts.append(hidden_for_output)
            combined = torch.cat(combined_parts, dim=1)
            x = self.layers(combined)
            output = x.squeeze(1)
            return output, hidden_next

        step_features = torch.cat((input, output, (input - output) / dt), dim=1)
        hidden_input = self.hidden_layers(step_features)

        if not self.has_hidden_state:
            hidden_for_output = input.new_zeros((input.shape[0], 0))
        elif self.core_type == "lstm":
            hidden_state, cell_state = hidden
            hidden_state, cell_state = self.recurrent_cell(hidden_input, (hidden_state, cell_state))
            hidden_for_output = hidden_state
            hidden = (hidden_state, cell_state)
        else:
            hidden_state = self.recurrent_cell(hidden_input, hidden)
            hidden_for_output = hidden_state
            hidden = hidden_state

        combined = torch.cat((input, output, (input - output) / dt, hidden_for_output), dim=1)
        x = self.layers(combined)
        output = x.squeeze(1)
        return output, hidden

    def initHidden(self, b_size: int, device: torch.device | None = None) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if device is None:
            device = next(self.parameters()).device
        if not self.has_hidden_state:
            return torch.zeros(b_size, 0, device=device)
        hidden = torch.zeros(b_size, self.hidden_size, device=device)
        if self.core_type == "lstm":
            return hidden.clone(), hidden.clone()
        return hidden


@dataclass
class ExperimentConfig:
    """Single config object shared by training, Optuna, and follow-up studies."""

    CORE_TYPE: str = "rnn"
    EPOCHS: int = 250
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-5
    GRAD_CLIP_VALUE: float = 1.0
    LR_FACTOR: float = 0.5
    LR_PATIENCE: int = 20
    EARLY_STOPPING_PATIENCE: int = 80
    MIN_EPOCHS: int = 100
    N_HIDDEN: int = 16
    READOUT_WIDTH: int = 128
    READOUT_DEPTH: int = 3
    FEATURE_WIDTH: int = 64
    FEATURE_DEPTH: int = 2
    SHUFFLE: bool = True
    NUM_WORKERS: int = 0
    SEED: int = DEFAULT_SPLIT_SEED
    MODEL_TAG: str = "baseline"
    USE_TRUE_INITIAL_OUTPUT: bool = True
    PAPER_USE_RATE_IN_STRESS: bool = False
    VERBOSE: bool = False
    PRINT_EVERY_EPOCHS: int = 10


def build_layer_sizes(
    input_dim: int,
    output_dim: int,
    n_hidden: int,
    readout_width: int,
    readout_depth: int,
    feature_width: int,
    feature_depth: int,
    core_type: str = "rnn",
    paper_use_rate_in_stress: bool = False,
) -> tuple[list[int], list[int]]:
    core_type = core_type.lower()
    if core_type == "baseline_rno":
        layer_input = [input_dim + 1 + n_hidden]
    elif core_type == "paper_rno":
        layer_input = [input_dim + n_hidden + (1 if paper_use_rate_in_stress else 0)]
    else:
        layer_input = [input_dim + output_dim + 1 + n_hidden]
    for _ in range(max(readout_depth - 1, 0)):
        layer_input.append(readout_width)
    layer_input.append(output_dim)

    if core_type in {"baseline_rno", "paper_rno"}:
        if n_hidden <= 0:
            layer_hidden = []
            return layer_input, layer_hidden
        hidden_input_size = input_dim + n_hidden
        if feature_depth <= 0:
            layer_hidden = [hidden_input_size, n_hidden]
        else:
            layer_hidden = [hidden_input_size]
            for _ in range(max(feature_depth - 1, 0)):
                layer_hidden.append(feature_width)
            layer_hidden.append(n_hidden)
    elif feature_depth <= 0:
        layer_hidden = [input_dim + output_dim + 1]
    else:
        layer_hidden = [input_dim + output_dim + 1]
        for _ in range(max(feature_depth - 1, 0)):
            layer_hidden.append(feature_width)
        layer_hidden.append(feature_width)
    return layer_input, layer_hidden


def build_net(config: ExperimentConfig, input_dim: int = 1, output_dim: int = 1) -> tuple[RNO, list[int], list[int]]:
    layer_input, layer_hidden = build_layer_sizes(
        input_dim=input_dim,
        output_dim=output_dim,
        n_hidden=config.N_HIDDEN,
        readout_width=config.READOUT_WIDTH,
        readout_depth=config.READOUT_DEPTH,
        feature_width=config.FEATURE_WIDTH,
        feature_depth=config.FEATURE_DEPTH,
        core_type=config.CORE_TYPE,
        paper_use_rate_in_stress=config.PAPER_USE_RATE_IN_STRESS,
    )
    net = RNO(
        input_size=input_dim,
        hidden_size=config.N_HIDDEN,
        output_size=output_dim,
        layer_input=layer_input,
        layer_hidden=layer_hidden,
        core_type=config.CORE_TYPE,
        paper_use_rate_in_stress=config.PAPER_USE_RATE_IN_STRESS,
    )
    return net, layer_input, layer_hidden


@dataclass
class InitialStressModel:
    slope: float
    intercept: float

    def predict(self, rate: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(rate, torch.Tensor):
            slope = torch.tensor(self.slope, device=rate.device, dtype=rate.dtype)
            intercept = torch.tensor(self.intercept, device=rate.device, dtype=rate.dtype)
            return slope * rate + intercept
        return self.slope * np.asarray(rate) + self.intercept

    def to_dict(self) -> dict[str, float]:
        return {"slope": self.slope, "intercept": self.intercept}

    @classmethod
    def from_dict(cls, payload: dict[str, float]) -> "InitialStressModel":
        return cls(slope=float(payload["slope"]), intercept=float(payload["intercept"]))


def fit_initial_stress_model(data_input: np.ndarray | torch.Tensor, data_output: np.ndarray | torch.Tensor, dt: float) -> InitialStressModel:
    data_input = to_numpy(data_input)
    data_output = to_numpy(data_output)
    rate0 = (data_input[:, 1] - data_input[:, 0]) / dt
    design = np.column_stack([rate0, np.ones_like(rate0)])
    coefficients, _, _, _ = np.linalg.lstsq(design, data_output[:, 0], rcond=None)
    return InitialStressModel(slope=float(coefficients[0]), intercept=float(coefficients[1]))


def save_initial_stress_model(initial_stress_model: InitialStressModel, path: str | Path) -> Path:
    return write_json(path, initial_stress_model.to_dict())


def load_initial_stress_model(path: str | Path) -> InitialStressModel:
    return InitialStressModel.from_dict(read_json(path))


def load_raw_data(
    train_path: str | Path = TRAIN_PATH,
    f_field: str = F_FIELD,
    sig_field: str = SIG_FIELD,
) -> tuple[np.ndarray, np.ndarray]:
    data_loader = MatReader(train_path, to_torch=False, to_float=True)
    data_input = np.asarray(data_loader.read_field(f_field), dtype=np.float32)
    data_output = np.asarray(data_loader.read_field(sig_field), dtype=np.float32)
    return data_input, data_output


def create_split_indices(
    n_total: int = Ntotal,
    train_size_value: int = train_size,
    val_size_value: int = val_size,
    seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, np.ndarray]:
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(n_total)
    train_idx = np.sort(permutation[:train_size_value])
    val_idx = np.sort(permutation[train_size_value : train_size_value + val_size_value])
    test_idx = np.sort(permutation[train_size_value + val_size_value :])
    return {"train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx}


def save_split_indices(split_indices: dict[str, np.ndarray], path: str | Path) -> Path:
    path = Path(path)
    ensure_directory(path.parent)
    np.savez(path, **split_indices)
    return path


def load_split_indices(path: str | Path) -> dict[str, np.ndarray]:
    loaded = np.load(path)
    return {key: loaded[key] for key in loaded.files}


def save_normalizers(
    input_normalizer: MinMaxNormalizer,
    output_normalizer: MinMaxNormalizer,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> tuple[Path, Path]:
    directories = ensure_artifact_tree(artifact_root)
    input_path = directories["normalization"] / "input_normalizer.json"
    output_path = directories["normalization"] / "output_normalizer.json"
    write_json(input_path, input_normalizer.to_dict())
    write_json(output_path, output_normalizer.to_dict())
    return input_path, output_path


def load_normalizers(artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT) -> tuple[MinMaxNormalizer, MinMaxNormalizer]:
    directories = ensure_artifact_tree(artifact_root)
    input_normalizer = MinMaxNormalizer.from_dict(read_json(directories["normalization"] / "input_normalizer.json"))
    output_normalizer = MinMaxNormalizer.from_dict(read_json(directories["normalization"] / "output_normalizer.json"))
    return input_normalizer, output_normalizer


def prepare_data(
    train_path: str | Path = TRAIN_PATH,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    split_seed: int = DEFAULT_SPLIT_SEED,
    f_field: str = F_FIELD,
    sig_field: str = SIG_FIELD,
) -> dict[str, Any]:
    """Load the raw histories, build a deterministic split, and fit train-only scalers."""
    configure_matplotlib_cache(artifact_root)
    directories = ensure_artifact_tree(artifact_root)

    data_input_raw, data_output_raw = load_raw_data(train_path=train_path, f_field=f_field, sig_field=sig_field)
    n_total_local = data_input_raw.shape[0]
    if n_total_local != Ntotal:
        raise ValueError(f"Expected Ntotal={Ntotal}, found {n_total_local}")

    split_indices = create_split_indices(n_total=n_total_local, seed=split_seed)
    split_path = save_split_indices(split_indices, directories["splits"] / f"split_seed_{split_seed}.npz")

    x_train_raw = data_input_raw[split_indices["train_idx"]]
    y_train_raw = data_output_raw[split_indices["train_idx"]]
    x_val_raw = data_input_raw[split_indices["val_idx"]]
    y_val_raw = data_output_raw[split_indices["val_idx"]]
    x_test_raw = data_input_raw[split_indices["test_idx"]]
    y_test_raw = data_output_raw[split_indices["test_idx"]]

    input_normalizer = MinMaxNormalizer().fit(x_train_raw)
    output_normalizer = MinMaxNormalizer().fit(y_train_raw)
    input_normalizer_path, output_normalizer_path = save_normalizers(
        input_normalizer=input_normalizer,
        output_normalizer=output_normalizer,
        artifact_root=artifact_root,
    )

    data_input = input_normalizer.transform(data_input_raw).astype(np.float32)
    data_output = output_normalizer.transform(data_output_raw).astype(np.float32)

    x_train = torch.from_numpy(data_input[split_indices["train_idx"]]).float()
    y_train = torch.from_numpy(data_output[split_indices["train_idx"]]).float()
    x_val = torch.from_numpy(data_input[split_indices["val_idx"]]).float()
    y_val = torch.from_numpy(data_output[split_indices["val_idx"]]).float()
    x_test = torch.from_numpy(data_input[split_indices["test_idx"]]).float()
    y_test = torch.from_numpy(data_output[split_indices["test_idx"]]).float()

    inputsize = data_input.shape[1]
    dt = 1.0 / (inputsize - 1)
    initial_stress_model = fit_initial_stress_model(x_train, y_train, dt)
    initial_stress_model_path = save_initial_stress_model(
        initial_stress_model,
        directories["normalization"] / "initial_stress_model.json",
    )

    return {
        "TRAIN_PATH": str(train_path),
        "F_FIELD": f_field,
        "SIG_FIELD": sig_field,
        "Ntotal": n_total_local,
        "train_size": train_size,
        "val_size": val_size,
        "test_start": test_start,
        "N_test": N_test,
        "dt": dt,
        "inputsize": inputsize,
        "data_input_raw": data_input_raw,
        "data_output_raw": data_output_raw,
        "data_input": data_input,
        "data_output": data_output,
        "x_train_raw": x_train_raw,
        "y_train_raw": y_train_raw,
        "x_val_raw": x_val_raw,
        "y_val_raw": y_val_raw,
        "x_test_raw": x_test_raw,
        "y_test_raw": y_test_raw,
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "x_test": x_test,
        "y_test": y_test,
        "split_indices": split_indices,
        "split_path": split_path,
        "input_normalizer": input_normalizer,
        "output_normalizer": output_normalizer,
        "input_normalizer_path": input_normalizer_path,
        "output_normalizer_path": output_normalizer_path,
        "initial_stress_model": initial_stress_model,
        "initial_stress_model_path": initial_stress_model_path,
    }


def rollout_sequence(
    net: RNO,
    x: torch.Tensor,
    dt: float,
    initial_stress_model: InitialStressModel,
    y_true0: torch.Tensor | None = None,
) -> torch.Tensor:
    """Roll the constitutive model forward one time step at a time over the full history."""
    batch_size, T = x.shape
    hidden = net.initHidden(batch_size, device=x.device)
    y_approx = torch.zeros(batch_size, T, device=x.device, dtype=x.dtype)

    if y_true0 is not None:
        # Training and evaluation can pin the first stress to the known target value.
        y_approx[:, 0] = y_true0
    else:
        # Free-running inference estimates the first stress from the fitted initial-rate model.
        rate0 = (x[:, 1] - x[:, 0]) / dt
        y_approx[:, 0] = initial_stress_model.predict(rate0)

    for i in range(1, T):
        y_approx[:, i], hidden = net(x[:, i].unsqueeze(1), x[:, i - 1].unsqueeze(1), hidden, dt)
    return y_approx


def compute_loss(loss_func: nn.Module, y_pred: torch.Tensor, y_true: torch.Tensor, start_index: int = 1) -> torch.Tensor:
    return loss_func(y_pred[:, start_index:], y_true[:, start_index:])


def compute_recurrent_grad_norm(net: RNO) -> float:
    total = 0.0
    for name, parameter in net.named_parameters():
        if "recurrent_cell" not in name or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().norm().item() ** 2)
    return total ** 0.5


def compute_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    output_normalizer: MinMaxNormalizer,
    start_index: int = 1,
) -> dict[str, float]:
    """Report metrics on raw stress values so tables stay physically interpretable."""
    y_true_np = to_numpy(y_true)[:, start_index:]
    y_pred_np = to_numpy(y_pred)[:, start_index:]
    y_true_raw = to_numpy(output_normalizer.inverse_transform(y_true_np))
    y_pred_raw = to_numpy(output_normalizer.inverse_transform(y_pred_np))

    truth = y_true_raw.reshape(-1)
    pred = y_pred_raw.reshape(-1)
    residual = pred - truth
    rmse = float(np.sqrt(mean_squared_error(truth, pred)))
    mae = float(mean_absolute_error(truth, pred))
    r2 = float(r2_score(truth, pred))
    truth_range = float(truth.max() - truth.min())
    truth_l2 = float(np.linalg.norm(truth))
    residual_l2 = float(np.linalg.norm(residual))
    nrmse = rmse / truth_range if truth_range > 1e-12 else 0.0
    relative_l2 = residual_l2 / truth_l2 if truth_l2 > 1e-12 else 0.0
    residual_mean = float(residual.mean())
    residual_std = float(residual.std())

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nrmse": nrmse,
        "relative_l2": relative_l2,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
    }


def make_data_loader(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool, num_workers: int) -> torch.utils.data.DataLoader:
    dataset = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def save_checkpoint(
    checkpoint_path: str | Path,
    net: RNO,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.ReduceLROnPlateau,
    config: ExperimentConfig,
    layer_input: list[int],
    layer_hidden: list[int],
    best_epoch: int,
    best_val_loss: float,
    initial_stress_model: InitialStressModel,
) -> Path:
    checkpoint_payload = {
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "config": asdict(config),
        "layer_input": layer_input,
        "layer_hidden": layer_hidden,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "initial_stress_model": initial_stress_model.to_dict(),
    }
    checkpoint_path = Path(checkpoint_path)
    ensure_directory(checkpoint_path.parent)
    torch.save(checkpoint_payload, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device | None = None,
) -> tuple[RNO, dict[str, Any]]:
    """Restore a saved model together with the config needed to rebuild its architecture."""
    if device is None:
        device = select_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = ExperimentConfig(**checkpoint["config"])
    net = RNO(
        input_size=1,
        hidden_size=config.N_HIDDEN,
        output_size=1,
        layer_input=checkpoint["layer_input"],
        layer_hidden=checkpoint["layer_hidden"],
        core_type=config.CORE_TYPE,
        paper_use_rate_in_stress=config.PAPER_USE_RATE_IN_STRESS,
    )
    net.load_state_dict(checkpoint["model_state_dict"])
    net.to(device)
    net.eval()
    return net, checkpoint


def train_model(
    data_bundle: dict[str, Any],
    config: ExperimentConfig,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_name: str | None = None,
    train_on_train_plus_val: bool = False,
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    """Run the full training loop, checkpoint the best epoch, and log every epoch to disk."""
    configure_matplotlib_cache(artifact_root)
    directories = ensure_artifact_tree(artifact_root)
    set_seeds(config.SEED)

    device = select_device()
    dt = float(data_bundle["dt"])
    loss_func = nn.MSELoss()

    if train_on_train_plus_val:
        # Final retraining mode combines the train and validation subsets after model selection.
        x_train = torch.cat([data_bundle["x_train"], data_bundle["x_val"]], dim=0)
        y_train = torch.cat([data_bundle["y_train"], data_bundle["y_val"]], dim=0)
        x_val = None
        y_val = None
        shuffle = True
    else:
        x_train = data_bundle["x_train"]
        y_train = data_bundle["y_train"]
        x_val = data_bundle["x_val"]
        y_val = data_bundle["y_val"]
        shuffle = config.SHUFFLE

    train_loader = make_data_loader(
        x=x_train,
        y=y_train,
        batch_size=config.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
    )

    input_dim = 1
    output_dim = 1
    net, layer_input, layer_hidden = build_net(config, input_dim=input_dim, output_dim=output_dim)
    net.to(device)

    optimizer = optim.AdamW(net.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.LR_FACTOR,
        patience=config.LR_PATIENCE,
    )

    initial_stress_model = data_bundle["initial_stress_model"]
    if run_name is None:
        run_name = f"{config.CORE_TYPE}_{config.MODEL_TAG}_seed_{config.SEED}"

    history_rows: list[dict[str, float | int]] = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    target_epochs = fixed_epochs if fixed_epochs is not None else config.EPOCHS
    checkpoint_path = directories["checkpoints"] / f"{run_name}.pt"
    run_start = time.perf_counter()

    if config.VERBOSE:
        print(
            f"[train] run={run_name} device={device} core={config.CORE_TYPE} "
            f"n_hidden={config.N_HIDDEN} batch={config.BATCH_SIZE} "
            f"epochs={target_epochs} lr={config.LEARNING_RATE:.3e}",
            flush=True,
        )

    for ep in range(1, target_epochs + 1):
        epoch_start = time.perf_counter()
        net.train()
        train_loss_accumulator = 0.0
        train_grad_norm_accumulator = 0.0
        n_train_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad(set_to_none=True)
            y_approx = rollout_sequence(
                net=net,
                x=x_batch,
                dt=dt,
                initial_stress_model=initial_stress_model,
                y_true0=y_batch[:, 0] if config.USE_TRUE_INITIAL_OUTPUT else None,
            )
            loss = compute_loss(loss_func=loss_func, y_pred=y_approx, y_true=y_batch)
            loss.backward()
            train_grad_norm_accumulator += compute_recurrent_grad_norm(net)
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=config.GRAD_CLIP_VALUE)
            optimizer.step()
            train_loss_accumulator += float(loss.item())
            n_train_batches += 1

        train_loss = train_loss_accumulator / max(n_train_batches, 1)
        train_grad_norm = train_grad_norm_accumulator / max(n_train_batches, 1)

        if x_val is not None and y_val is not None:
            # Standard training mode uses the held-out validation split for model selection.
            net.eval()
            with torch.no_grad():
                y_val_device = y_val.to(device)
                x_val_device = x_val.to(device)
                y_val_approx = rollout_sequence(
                    net=net,
                    x=x_val_device,
                    dt=dt,
                    initial_stress_model=initial_stress_model,
                    y_true0=y_val_device[:, 0] if config.USE_TRUE_INITIAL_OUTPUT else None,
                )
                val_loss = float(compute_loss(loss_func=loss_func, y_pred=y_val_approx, y_true=y_val_device).item())
                val_metrics = compute_metrics(
                    y_true=y_val_device,
                    y_pred=y_val_approx,
                    output_normalizer=data_bundle["output_normalizer"],
                )
        else:
            # Retraining mode reuses the combined training set only for monitored bookkeeping.
            net.eval()
            with torch.no_grad():
                x_train_device = x_train.to(device)
                y_train_device = y_train.to(device)
                y_train_approx = rollout_sequence(
                    net=net,
                    x=x_train_device,
                    dt=dt,
                    initial_stress_model=initial_stress_model,
                    y_true0=y_train_device[:, 0] if config.USE_TRUE_INITIAL_OUTPUT else None,
                )
                val_loss = float(compute_loss(loss_func=loss_func, y_pred=y_train_approx, y_true=y_train_device).item())
                val_metrics = compute_metrics(
                    y_true=y_train_device,
                    y_pred=y_train_approx,
                    output_normalizer=data_bundle["output_normalizer"],
                )

        scheduler.step(val_loss)

        row = {
            "epoch": ep,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_grad_norm": train_grad_norm,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history_rows.append(row)

        if val_loss < best_val_loss - 1e-8:
            # Persist only the best-performing checkpoint so later analysis always loads it directly.
            best_val_loss = val_loss
            best_epoch = ep
            epochs_without_improvement = 0
            save_checkpoint(
                checkpoint_path=checkpoint_path,
                net=net,
                optimizer=optimizer,
                scheduler=scheduler,
                config=config,
                layer_input=layer_input,
                layer_hidden=layer_hidden,
                best_epoch=best_epoch,
                best_val_loss=best_val_loss,
                initial_stress_model=initial_stress_model,
            )
        else:
            epochs_without_improvement += 1

        if config.VERBOSE:
            should_print = (
                ep == 1
                or ep == target_epochs
                or ep % max(config.PRINT_EVERY_EPOCHS, 1) == 0
                or epochs_without_improvement == 0
            )
            if should_print:
                epoch_seconds = time.perf_counter() - epoch_start
                print(
                    f"[train] run={run_name} epoch={ep}/{target_epochs} "
                    f"train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
                    f"best_val={best_val_loss:.6e} lr={optimizer.param_groups[0]['lr']:.3e} "
                    f"grad_norm={train_grad_norm:.3e} epoch_s={epoch_seconds:.1f}",
                    flush=True,
                )

        if fixed_epochs is None and ep >= config.MIN_EPOCHS and epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            if config.VERBOSE:
                print(
                    f"[train] run={run_name} early_stop epoch={ep} "
                    f"best_epoch={best_epoch} best_val={best_val_loss:.6e}",
                    flush=True,
                )
            break

    total_seconds = time.perf_counter() - run_start
    history = pd.DataFrame(history_rows)
    history_path = directories["logs"] / f"{run_name}_history.csv"
    history.to_csv(history_path, index=False)

    config_path = directories["logs"] / f"{run_name}_config.json"
    write_json(config_path, asdict(config))

    summary_path = directories["logs"] / f"{run_name}_summary.json"
    summary_payload = {
        "run_name": run_name,
        "checkpoint_path": str(checkpoint_path),
        "history_path": str(history_path),
        "config_path": str(config_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "device": str(device),
        "n_epochs_completed": int(history.shape[0]),
        "total_seconds": float(total_seconds),
    }
    write_json(summary_path, summary_payload)

    if config.VERBOSE:
        print(
            f"[train] run={run_name} complete epochs={history.shape[0]} "
            f"best_epoch={best_epoch} best_val={best_val_loss:.6e} total_s={total_seconds:.1f} "
            f"checkpoint={checkpoint_path}",
            flush=True,
        )

    return {
        **summary_payload,
        "history": history,
        "summary_path": summary_path,
    }


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    x: torch.Tensor,
    y: torch.Tensor,
    output_normalizer: MinMaxNormalizer,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_name: str = "evaluation",
    y_true0: torch.Tensor | None = None,
) -> dict[str, Any]:
    device = select_device()
    net, checkpoint = load_checkpoint(checkpoint_path, device=device)
    initial_stress_model = InitialStressModel.from_dict(checkpoint["initial_stress_model"])
    dt = 1.0 / (x.shape[1] - 1)

    with torch.no_grad():
        x_device = x.to(device)
        y_device = y.to(device)
        y_pred = rollout_sequence(
            net=net,
            x=x_device,
            dt=dt,
            initial_stress_model=initial_stress_model,
            y_true0=y_true0.to(device) if y_true0 is not None else None,
        )

    metrics = compute_metrics(y_true=y_device, y_pred=y_pred, output_normalizer=output_normalizer)
    directories = ensure_artifact_tree(artifact_root)
    prediction_path = directories["predictions"] / f"{run_name}_predictions.npz"
    np.savez(
        prediction_path,
        y_true=to_numpy(y_device),
        y_pred=to_numpy(y_pred),
        y_true_raw=to_numpy(output_normalizer.inverse_transform(y_device)),
        y_pred_raw=to_numpy(output_normalizer.inverse_transform(y_pred)),
    )
    return {
        "metrics": metrics,
        "prediction_path": prediction_path,
        "checkpoint": checkpoint,
    }


def make_residual_arrays(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    output_normalizer: MinMaxNormalizer,
    start_index: int = 1,
) -> dict[str, np.ndarray]:
    y_true_raw = to_numpy(output_normalizer.inverse_transform(y_true))[:, start_index:]
    y_pred_raw = to_numpy(output_normalizer.inverse_transform(y_pred))[:, start_index:]
    residual_raw = y_pred_raw - y_true_raw
    return {
        "y_true_raw": y_true_raw,
        "y_pred_raw": y_pred_raw,
        "residual_raw": residual_raw,
    }


def triangular_wave(t: np.ndarray, amplitude: float, cycles: int) -> np.ndarray:
    phase = (cycles * t) % 1.0
    return amplitude * (4.0 * np.abs(phase - 0.5) - 1.0)


def generate_unseen_load_cases(num_points: int = 1001) -> dict[str, np.ndarray]:
    t = np.linspace(0.0, 1.0, num_points, dtype=np.float32)
    unseen_cases = {
        "monotonic_ramp": 0.18 * t,
        "ramp_hold_relaxation": np.where(t < 0.35, 0.16 * (t / 0.35), 0.16),
        "triangular_cycle": triangular_wave(t, amplitude=0.15, cycles=2).astype(np.float32),
        "sinusoidal_shifted": (0.12 * np.sin(2.0 * np.pi * 3.0 * t) + 0.03 * np.sin(2.0 * np.pi * 7.0 * t)).astype(np.float32),
    }
    return {key: value.astype(np.float32) for key, value in unseen_cases.items()}


def plot_sample_histories(data_input_raw: np.ndarray, data_output_raw: np.ndarray, save_path: str | Path, sample_count: int = 6) -> Path:
    indices = np.linspace(0, data_input_raw.shape[0] - 1, sample_count, dtype=int)
    time_axis = np.linspace(0.0, 1.0, data_input_raw.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for idx in indices:
        axes[0].plot(time_axis, data_input_raw[idx], linewidth=1.0, label=f"sample {idx}")
        axes[1].plot(time_axis, data_output_raw[idx], linewidth=1.0, label=f"sample {idx}")
    axes[0].set_title("Representative macroscopic strain histories")
    axes[0].set_ylabel("strain")
    axes[1].set_title("Representative macroscopic stress histories")
    axes[1].set_ylabel("stress")
    axes[1].set_xlabel("time")
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_distributions(data_input_raw: np.ndarray, data_output_raw: np.ndarray, save_path: str | Path) -> Path:
    rate = np.diff(data_input_raw, axis=1) * (data_input_raw.shape[1] - 1)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(data_input_raw.reshape(-1), bins=60, color="#1f77b4", alpha=0.85)
    axes[0].set_title("Strain distribution")
    axes[1].hist(data_output_raw.reshape(-1), bins=60, color="#d62728", alpha=0.85)
    axes[1].set_title("Stress distribution")
    axes[2].hist(rate.reshape(-1), bins=60, color="#2ca02c", alpha=0.85)
    axes[2].set_title("Strain-rate distribution")
    for axis in axes:
        axis.grid(alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_strain_stress_scatter(
    data_input_raw: np.ndarray,
    data_output_raw: np.ndarray,
    save_path: str | Path,
    scatter_samples: int = 20000,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Path:
    generator = np.random.default_rng(seed)
    flat_strain = data_input_raw.reshape(-1)
    flat_stress = data_output_raw.reshape(-1)
    choice = generator.choice(flat_strain.size, size=min(scatter_samples, flat_strain.size), replace=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(flat_strain[choice], flat_stress[choice], s=8, alpha=0.2, color="#9467bd")
    ax.set_title("Subsampled strain-stress relationship")
    ax.set_xlabel("strain")
    ax.set_ylabel("stress")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_normalization_comparison(
    data_input_raw: np.ndarray,
    data_output_raw: np.ndarray,
    input_normalizer: MinMaxNormalizer,
    output_normalizer: MinMaxNormalizer,
    save_path: str | Path,
) -> Path:
    sample_index = int(np.argmax(np.std(data_output_raw, axis=1)))
    zscore_input = ZScoreNormalizer().fit(data_input_raw).transform(data_input_raw[sample_index])
    zscore_output = ZScoreNormalizer().fit(data_output_raw).transform(data_output_raw[sample_index])
    minmax_input = to_numpy(input_normalizer.transform(data_input_raw[sample_index]))
    minmax_output = to_numpy(output_normalizer.transform(data_output_raw[sample_index]))
    time_axis = np.linspace(0.0, 1.0, data_input_raw.shape[1])

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(time_axis, data_input_raw[sample_index], label="raw", linewidth=1.3)
    axes[0].plot(time_axis, minmax_input, label="min-max", linewidth=1.0)
    axes[0].plot(time_axis, zscore_input, label="z-score", linewidth=1.0)
    axes[0].set_title(f"Normalization comparison for strain sample {sample_index}")
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(time_axis, data_output_raw[sample_index], label="raw", linewidth=1.3)
    axes[1].plot(time_axis, minmax_output, label="min-max", linewidth=1.0)
    axes[1].plot(time_axis, zscore_output, label="z-score", linewidth=1.0)
    axes[1].set_title(f"Normalization comparison for stress sample {sample_index}")
    axes[1].set_ylabel("value")
    axes[1].set_xlabel("time")
    axes[1].grid(alpha=0.3)
    axes[1].legend()
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_acf_pacf(
    data_input_raw: np.ndarray,
    data_output_raw: np.ndarray,
    save_path: str | Path,
    lags: int = 60,
) -> Path | None:
    if not HAS_STATSMODELS:
        return None

    strain_energy = np.mean(np.abs(data_input_raw), axis=1)
    sample_index = int(np.argmin(np.abs(strain_energy - np.median(strain_energy))))
    strain_series = data_input_raw[sample_index]
    stress_series = data_output_raw[sample_index]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].stem(acf(strain_series, nlags=lags), basefmt=" ")
    axes[0, 0].set_title(f"Strain ACF (sample {sample_index})")
    axes[0, 1].stem(pacf(strain_series, nlags=min(lags, len(strain_series) // 2 - 1)), basefmt=" ")
    axes[0, 1].set_title(f"Strain PACF (sample {sample_index})")
    axes[1, 0].stem(acf(stress_series, nlags=lags), basefmt=" ")
    axes[1, 0].set_title(f"Stress ACF (sample {sample_index})")
    axes[1, 1].stem(pacf(stress_series, nlags=min(lags, len(stress_series) // 2 - 1)), basefmt=" ")
    axes[1, 1].set_title(f"Stress PACF (sample {sample_index})")

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.3)

    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def stationarity_summary(data_input_raw: np.ndarray, data_output_raw: np.ndarray) -> pd.DataFrame:
    if not HAS_STATSMODELS:
        return pd.DataFrame(
            [{"series": "statsmodels_unavailable", "adf_statistic": np.nan, "p_value": np.nan, "note": "Install statsmodels"}]
        )

    sample_index = int(np.argmax(np.std(data_output_raw, axis=1)))
    rows = []
    for label, series in [("strain", data_input_raw[sample_index]), ("stress", data_output_raw[sample_index])]:
        adf_statistic, p_value, *_ = adfuller(series)
        rows.append(
            {
                "series": f"{label}_sample_{sample_index}",
                "adf_statistic": float(adf_statistic),
                "p_value": float(p_value),
                "note": "Exploratory only for driven response data",
            }
        )
    return pd.DataFrame(rows)


def run_environment_and_mps_checks(
    TRAIN_PATH: str | Path = TRAIN_PATH,
    ARTIFACT_ROOT: str | Path = DEFAULT_ARTIFACT_ROOT,
    SEED: int = DEFAULT_SPLIT_SEED,
    MPS_TEST_BATCH: int = 4,
    MPS_TEST_SEQ_LEN: int = 1001,
    MPS_TEST_HIDDEN: int = 16,
) -> dict[str, Any]:
    configure_matplotlib_cache(ARTIFACT_ROOT)
    directories = ensure_artifact_tree(ARTIFACT_ROOT)
    set_seeds(SEED)

    summary_rows: list[dict[str, Any]] = []
    summary_rows.append({"item": "python_executable", "value": sys.executable})
    summary_rows.append({"item": "python_version", "value": platform.python_version()})
    summary_rows.append({"item": "platform", "value": platform.platform()})
    summary_rows.append({"item": "torch_version", "value": torch.__version__})
    summary_rows.append({"item": "mps_built", "value": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())})
    summary_rows.append({"item": "mps_available", "value": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())})
    summary_rows.append({"item": "cuda_available", "value": torch.cuda.is_available()})
    summary_rows.append({"item": "selected_device", "value": str(select_device())})

    data_input_raw, data_output_raw = load_raw_data(train_path=TRAIN_PATH)
    summary_rows.append({"item": "data_input_shape", "value": list(data_input_raw.shape)})
    summary_rows.append({"item": "data_output_shape", "value": list(data_output_raw.shape)})
    summary_rows.append({"item": "data_input_min", "value": float(data_input_raw.min())})
    summary_rows.append({"item": "data_input_max", "value": float(data_input_raw.max())})
    summary_rows.append({"item": "data_output_min", "value": float(data_output_raw.min())})
    summary_rows.append({"item": "data_output_max", "value": float(data_output_raw.max())})

    smoke_rows: list[dict[str, Any]] = []
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        dt = 1.0 / max(MPS_TEST_SEQ_LEN - 1, 1)

        def record(name: str, ok: bool, detail: str) -> None:
            smoke_rows.append({"test": name, "ok": ok, "detail": detail})

        try:
            ff = nn.Sequential(nn.Linear(3, 32), nn.SELU(), nn.Linear(32, 1)).to(device)
            x_ff = torch.randn(MPS_TEST_BATCH, 3, device=device)
            y_ff = torch.randn(MPS_TEST_BATCH, 1, device=device)
            optimizer = optim.AdamW(ff.parameters(), lr=1e-3)
            loss_func = nn.MSELoss()
            optimizer.zero_grad(set_to_none=True)
            loss = loss_func(ff(x_ff), y_ff)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ff.parameters(), max_norm=1.0)
            optimizer.step()
            record("linear_selu_adamw", True, f"loss={float(loss.detach().cpu()):.6f}")
        except Exception as exc:
            record("linear_selu_adamw", False, repr(exc))

        for core_type in ["rnn", "gru", "lstm"]:
            try:
                config = ExperimentConfig(CORE_TYPE=core_type, N_HIDDEN=MPS_TEST_HIDDEN, EPOCHS=1, MIN_EPOCHS=1)
                net, _, _ = build_net(config)
                net.to(device)
                x = torch.randn(MPS_TEST_BATCH, MPS_TEST_SEQ_LEN, device=device)
                initial_stress_model = InitialStressModel(slope=1.0, intercept=0.0)
                y_pred = rollout_sequence(net=net, x=x, dt=dt, initial_stress_model=initial_stress_model, y_true0=None)
                y_true = torch.randn(MPS_TEST_BATCH, MPS_TEST_SEQ_LEN, device=device)
                optimizer = optim.AdamW(net.parameters(), lr=1e-3)
                optimizer.zero_grad(set_to_none=True)
                loss = compute_loss(nn.MSELoss(), y_pred, y_true)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
                scheduler.step(float(loss.detach().cpu()))
                record(f"{core_type}_explicit_rollout", True, f"loss={float(loss.detach().cpu()):.6f}")
            except Exception as exc:
                record(f"{core_type}_explicit_rollout", False, repr(exc))
    else:
        smoke_rows.append({"test": "mps_unavailable", "ok": False, "detail": "MPS is not available in this runtime"})

    summary_path = directories["environment"] / "environment_summary.json"
    smoke_path = directories["environment"] / "mps_smoke_tests.json"
    write_json(summary_path, {"summary_rows": summary_rows})
    write_json(smoke_path, {"smoke_rows": smoke_rows})

    return {
        "summary_rows": summary_rows,
        "smoke_rows": smoke_rows,
        "summary_path": summary_path,
        "smoke_path": smoke_path,
        "data_summary": {
            "Ntotal": int(data_input_raw.shape[0]),
            "inputsize": int(data_input_raw.shape[1]),
            "dt": float(1.0 / (data_input_raw.shape[1] - 1)),
        },
    }


def run_eda_and_preprocessing(
    TRAIN_PATH: str | Path = TRAIN_PATH,
    ARTIFACT_ROOT: str | Path = DEFAULT_ARTIFACT_ROOT,
    F_FIELD: str = F_FIELD,
    SIG_FIELD: str = SIG_FIELD,
    SEED: int = DEFAULT_SPLIT_SEED,
    ACF_LAGS: int = 60,
    SCATTER_SAMPLES: int = 20000,
    REPRESENTATIVE_SAMPLE_COUNT: int = 6,
) -> dict[str, Any]:
    configure_matplotlib_cache(ARTIFACT_ROOT)
    directories = ensure_artifact_tree(ARTIFACT_ROOT)
    set_seeds(SEED)

    data_bundle = prepare_data(
        train_path=TRAIN_PATH,
        artifact_root=ARTIFACT_ROOT,
        split_seed=SEED,
        f_field=F_FIELD,
        sig_field=SIG_FIELD,
    )

    data_input_raw = data_bundle["data_input_raw"]
    data_output_raw = data_bundle["data_output_raw"]
    input_normalizer = data_bundle["input_normalizer"]
    output_normalizer = data_bundle["output_normalizer"]

    summary_rows = [
        {"item": "Ntotal", "value": int(data_bundle["Ntotal"])},
        {"item": "train_size", "value": int(train_size)},
        {"item": "val_size", "value": int(val_size)},
        {"item": "test_size", "value": int(N_test)},
        {"item": "inputsize", "value": int(data_bundle["inputsize"])},
        {"item": "dt", "value": float(data_bundle["dt"])},
        {"item": "strain_min", "value": float(data_input_raw.min())},
        {"item": "strain_max", "value": float(data_input_raw.max())},
        {"item": "stress_min", "value": float(data_output_raw.min())},
        {"item": "stress_max", "value": float(data_output_raw.max())},
    ]

    split_rows = [
        {"split": "train", "n_samples": int(len(data_bundle["split_indices"]["train_idx"])), "indices_path": str(data_bundle["split_path"])},
        {"split": "validation", "n_samples": int(len(data_bundle["split_indices"]["val_idx"])), "indices_path": str(data_bundle["split_path"])},
        {"split": "test", "n_samples": int(len(data_bundle["split_indices"]["test_idx"])), "indices_path": str(data_bundle["split_path"])},
    ]

    normalization_rows = [
        {
            "normalizer": "input_minmax",
            "data_min": float(input_normalizer.data_min),
            "data_max": float(input_normalizer.data_max),
            "feature_min": float(input_normalizer.feature_min),
            "feature_max": float(input_normalizer.feature_max),
        },
        {
            "normalizer": "output_minmax",
            "data_min": float(output_normalizer.data_min),
            "data_max": float(output_normalizer.data_max),
            "feature_min": float(output_normalizer.feature_min),
            "feature_max": float(output_normalizer.feature_max),
        },
    ]

    figures = [
        plot_sample_histories(
            data_input_raw=data_input_raw,
            data_output_raw=data_output_raw,
            save_path=directories["figures"] / "01_sample_histories.png",
            sample_count=REPRESENTATIVE_SAMPLE_COUNT,
        ),
        plot_distributions(
            data_input_raw=data_input_raw,
            data_output_raw=data_output_raw,
            save_path=directories["figures"] / "01_distributions.png",
        ),
        plot_strain_stress_scatter(
            data_input_raw=data_input_raw,
            data_output_raw=data_output_raw,
            save_path=directories["figures"] / "01_scatter.png",
            scatter_samples=SCATTER_SAMPLES,
            seed=SEED,
        ),
        plot_normalization_comparison(
            data_input_raw=data_input_raw,
            data_output_raw=data_output_raw,
            input_normalizer=input_normalizer,
            output_normalizer=output_normalizer,
            save_path=directories["figures"] / "01_normalization_comparison.png",
        ),
    ]
    acf_figure = plot_acf_pacf(
        data_input_raw=data_input_raw,
        data_output_raw=data_output_raw,
        save_path=directories["figures"] / "01_acf_pacf.png",
        lags=ACF_LAGS,
    )
    if acf_figure is not None:
        figures.append(acf_figure)

    stationarity_table = stationarity_summary(data_input_raw=data_input_raw, data_output_raw=data_output_raw)
    stationarity_path = directories["eda"] / "stationarity_summary.csv"
    stationarity_table.to_csv(stationarity_path, index=False)

    summary_path = directories["eda"] / "eda_summary.csv"
    split_path = directories["eda"] / "split_summary.csv"
    normalization_path = directories["eda"] / "normalization_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(split_rows).to_csv(split_path, index=False)
    pd.DataFrame(normalization_rows).to_csv(normalization_path, index=False)

    metadata_path = directories["eda"] / "eda_metadata.json"
    write_json(
        metadata_path,
        {
            "summary_path": str(summary_path),
            "split_path": str(split_path),
            "normalization_path": str(normalization_path),
            "stationarity_path": str(stationarity_path),
            "figure_paths": [str(path) for path in figures],
            "split_seed": SEED,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": N_test,
        },
    )

    return {
        "summary_rows": summary_rows,
        "split_rows": split_rows,
        "normalization_rows": normalization_rows,
        "stationarity_rows": stationarity_table.to_dict(orient="records"),
        "summary_path": summary_path,
        "split_summary_path": split_path,
        "normalization_path": normalization_path,
        "stationarity_path": stationarity_path,
        "metadata_path": metadata_path,
        "figure_paths": figures,
        "data_bundle": data_bundle,
    }


def summarize_hidden_results(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame(
            columns=["n_hidden", "mean_val_loss", "std_val_loss", "median_best_epoch", "n_runs"]
        )

    return (
        results_df.groupby("n_hidden", as_index=False)
        .agg(
            mean_val_loss=("best_val_loss", "mean"),
            std_val_loss=("best_val_loss", "std"),
            median_best_epoch=("best_epoch", "median"),
            n_runs=("seed", "nunique"),
        )
        .sort_values("n_hidden")
    )


def hidden_threshold_from_results(
    results_df: pd.DataFrame,
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    require_plateau: bool | None = None,
) -> dict[str, Any]:
    grouped = summarize_hidden_results(results_df=results_df)
    if grouped.empty:
        raise ValueError("No hidden-threshold results are available.")

    if require_plateau is None:
        require_plateau = reference_loss is None

    best_loss = float(grouped["mean_val_loss"].min()) if reference_loss is None else float(reference_loss)
    threshold_limit = best_loss * (1.0 + tolerance_ratio)
    within_threshold = grouped["mean_val_loss"] <= threshold_limit
    grouped = grouped.assign(within_threshold=within_threshold.to_numpy())

    best_hidden = int(grouped.loc[grouped["mean_val_loss"].idxmin(), "n_hidden"])
    acceptable_hidden = [int(value) for value in grouped.loc[within_threshold, "n_hidden"].tolist()]

    if not acceptable_hidden:
        minimum_hidden = None
        selected_hidden = best_hidden
    elif require_plateau:
        plateau_candidates: list[int] = []
        for row_index, n_hidden_value in enumerate(grouped["n_hidden"].tolist()):
            if within_threshold.iloc[row_index] and bool(within_threshold.iloc[row_index:].all()):
                plateau_candidates.append(int(n_hidden_value))
        minimum_hidden = plateau_candidates[0] if plateau_candidates else acceptable_hidden[0]
        selected_hidden = minimum_hidden
    else:
        minimum_hidden = acceptable_hidden[0]
        selected_hidden = minimum_hidden

    return {
        "grouped_results": grouped,
        "best_loss": best_loss,
        "best_hidden": best_hidden,
        "threshold_limit": threshold_limit,
        "minimum_hidden": minimum_hidden,
        "selected_hidden": selected_hidden,
        "threshold_found": minimum_hidden is not None,
        "acceptable_hidden": acceptable_hidden,
        "reference_loss": reference_loss,
    }


def plot_hidden_threshold(
    results_df: pd.DataFrame,
    save_path: str | Path,
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    require_plateau: bool | None = None,
) -> tuple[Path, dict[str, Any]]:
    threshold_summary = hidden_threshold_from_results(
        results_df=results_df,
        tolerance_ratio=tolerance_ratio,
        reference_loss=reference_loss,
        require_plateau=require_plateau,
    )
    grouped = threshold_summary["grouped_results"]

    fig, ax = plt.subplots(figsize=(8, 5))
    raw_results = results_df.loc[:, ["n_hidden", "best_val_loss"]].copy()
    if not raw_results.empty:
        raw_results = raw_results.sort_values(["n_hidden", "best_val_loss"]).reset_index(drop=True)
        jitter_rng = np.random.default_rng(0)
        jitter = jitter_rng.uniform(-0.3, 0.3, size=len(raw_results))
        ax.scatter(
            raw_results["n_hidden"] + jitter,
            raw_results["best_val_loss"],
            alpha=0.35,
            s=24,
            color="#7f7f7f",
            label="individual seed runs",
        )
    ax.errorbar(grouped["n_hidden"], grouped["mean_val_loss"], yerr=grouped["std_val_loss"].fillna(0.0), marker="o", capsize=4)
    threshold_label = "5% plateau threshold" if threshold_summary["reference_loss"] is None else "5% target threshold"
    ax.axhline(threshold_summary["threshold_limit"], color="#d62728", linestyle="--", label=threshold_label)
    if threshold_summary["minimum_hidden"] is not None:
        ax.axvline(
            threshold_summary["minimum_hidden"],
            color="#2ca02c",
            linestyle=":",
            label=f"minimum hidden = {threshold_summary['minimum_hidden']}",
        )
    else:
        ax.axvline(
            threshold_summary["best_hidden"],
            color="#ff7f0e",
            linestyle=":",
            label=f"best tested hidden = {threshold_summary['best_hidden']}",
        )
    ax.set_title("Hidden-state threshold analysis")
    ax.set_xlabel("n_hidden")
    ax.set_ylabel("mean validation loss")
    ax.set_xticks(grouped["n_hidden"].tolist())
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path, threshold_summary


def load_hidden_threshold_cache(results_path: str | Path) -> pd.DataFrame:
    results_path = Path(results_path)
    if not results_path.exists():
        return pd.DataFrame(
            columns=[
                "n_hidden",
                "seed",
                "best_val_loss",
                "best_epoch",
                "checkpoint_path",
                "device",
                "search_stage",
                "search_order",
            ]
        )

    results_df = pd.read_csv(results_path)
    for column in ["device", "search_stage", "search_order"]:
        if column not in results_df.columns:
            results_df[column] = np.nan
    return results_df


def save_hidden_threshold_cache(results_df: pd.DataFrame, results_path: str | Path) -> Path:
    results_path = Path(results_path)
    ensure_directory(results_path.parent)
    sort_columns = [column for column in ["n_hidden", "seed"] if column in results_df.columns]
    if sort_columns:
        results_df = results_df.sort_values(sort_columns)
    results_df.to_csv(results_path, index=False)
    return results_path


def evaluate_hidden_size_cached(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    n_hidden: int,
    seeds: list[int],
    artifact_root: str | Path,
    run_prefix: str,
    results_path: str | Path,
    search_stage: str,
    search_order: int,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    results_df = load_hidden_threshold_cache(results_path)
    n_hidden = int(n_hidden)

    existing_rows = results_df.loc[results_df["n_hidden"] == n_hidden].copy()
    if not existing_rows.empty:
        missing_stage = existing_rows["search_stage"].isna()
        if missing_stage.any():
            results_df.loc[existing_rows.index[missing_stage], "search_stage"] = search_stage
        missing_order = existing_rows["search_order"].isna()
        if missing_order.any():
            results_df.loc[existing_rows.index[missing_order], "search_order"] = search_order

    for seed in seeds:
        already_done = bool(
            ((results_df["n_hidden"] == n_hidden) & (results_df["seed"] == seed)).any()
        )
        if already_done:
            if verbose:
                print(
                    f"[hidden] reuse core={base_config.CORE_TYPE} n_hidden={n_hidden} seed={seed} stage={search_stage}",
                    flush=True,
                )
            continue

        if verbose:
            print(
                f"[hidden] train core={base_config.CORE_TYPE} n_hidden={n_hidden} seed={seed} stage={search_stage}",
                flush=True,
            )
        config = copy.deepcopy(base_config)
        config.N_HIDDEN = n_hidden
        config.SEED = seed
        config.MODEL_TAG = f"{run_prefix}_h{n_hidden}"
        result = train_model(
            data_bundle=data_bundle,
            config=config,
            artifact_root=artifact_root,
            run_name=f"{run_prefix}_{config.CORE_TYPE}_h{n_hidden}_seed_{seed}",
        )
        results_df = pd.concat(
            [
                results_df,
                pd.DataFrame(
                    [
                        {
                            "n_hidden": n_hidden,
                            "seed": seed,
                            "best_val_loss": result["best_val_loss"],
                            "best_epoch": result["best_epoch"],
                            "checkpoint_path": result["checkpoint_path"],
                            "device": result.get("device"),
                            "search_stage": search_stage,
                            "search_order": search_order,
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        save_hidden_threshold_cache(results_df=results_df, results_path=results_path)
        if verbose:
            print(
                f"[hidden] done core={base_config.CORE_TYPE} n_hidden={n_hidden} seed={seed} "
                f"best_val={result['best_val_loss']:.6e} best_epoch={result['best_epoch']}",
                flush=True,
            )

    grouped = summarize_hidden_results(results_df=results_df)
    row = grouped.loc[grouped["n_hidden"] == n_hidden]
    if row.empty:
        raise RuntimeError(f"Hidden-size evaluation failed for n_hidden={n_hidden}.")
    return results_df, row.iloc[0].to_dict()


def run_hidden_threshold_sweep(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    hidden_grid: list[int],
    seeds: list[int],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_prefix: str = "hidden_threshold",
) -> dict[str, Any]:
    """Exhaustive hidden-size sweep over a fixed grid and a fixed seed list."""
    rows: list[dict[str, Any]] = []
    for n_hidden in hidden_grid:
        for seed in seeds:
            config = copy.deepcopy(base_config)
            config.N_HIDDEN = n_hidden
            config.SEED = seed
            config.MODEL_TAG = f"{run_prefix}_h{n_hidden}"
            result = train_model(
                data_bundle=data_bundle,
                config=config,
                artifact_root=artifact_root,
                run_name=f"{run_prefix}_{config.CORE_TYPE}_h{n_hidden}_seed_{seed}",
            )
            rows.append(
                {
                    "n_hidden": n_hidden,
                    "seed": seed,
                    "best_val_loss": result["best_val_loss"],
                    "best_epoch": result["best_epoch"],
                    "checkpoint_path": result["checkpoint_path"],
                }
            )

    results_df = pd.DataFrame(rows)
    directories = ensure_artifact_tree(artifact_root)
    results_path = directories["final"] / f"{run_prefix}_results.csv"
    results_df.to_csv(results_path, index=False)
    figure_path, threshold_summary = plot_hidden_threshold(
        results_df=results_df,
        save_path=directories["figures"] / f"{run_prefix}_threshold.png",
    )
    grouped_results_path = directories["final"] / f"{run_prefix}_grouped_results.csv"
    threshold_summary["grouped_results"].to_csv(grouped_results_path, index=False)
    threshold_json_path = directories["final"] / f"{run_prefix}_threshold.json"
    write_json(
        threshold_json_path,
        {
            "best_loss": threshold_summary["best_loss"],
            "threshold_limit": threshold_summary["threshold_limit"],
            "minimum_hidden": threshold_summary["minimum_hidden"],
            "grouped_results_path": str(grouped_results_path),
        },
    )
    return {
        "results_df": results_df,
        "results_path": results_path,
        "figure_path": figure_path,
        "grouped_results_path": grouped_results_path,
        "threshold_summary": threshold_summary,
        "threshold_json_path": threshold_json_path,
    }


def run_hidden_threshold_adaptive(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    seeds: list[int],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_prefix: str = "hidden_threshold",
    start_hidden: int = 4,
    minimum_hidden: int = 1,
    maximum_hidden: int = 48,
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    if reference_loss is None:
        raise ValueError("reference_loss must be provided for the adaptive hidden-threshold search.")

    directories = ensure_artifact_tree(artifact_root)
    results_path = directories["final"] / f"{run_prefix}_results.csv"
    trace_path = directories["final"] / f"{run_prefix}_trace.csv"

    search_rows: list[dict[str, Any]] = []
    evaluated_hidden: set[int] = set()
    search_order = 0

    threshold_limit = float(reference_loss) * (1.0 + tolerance_ratio)
    start_hidden = int(min(max(start_hidden, minimum_hidden), maximum_hidden))

    if verbose:
        print(
            f"[hidden] adaptive start seeds={seeds} start_hidden={start_hidden} "
            f"bounds=[{minimum_hidden}, {maximum_hidden}] threshold={threshold_limit:.6e}",
            flush=True,
        )

    def evaluate(n_hidden: int, search_stage: str) -> tuple[dict[str, Any], bool]:
        nonlocal search_order
        clamped_hidden = int(min(max(n_hidden, minimum_hidden), maximum_hidden))
        if verbose:
            print(
                f"[hidden] evaluate n_hidden={clamped_hidden} stage={search_stage} "
                f"seeds={seeds} order={search_order}",
                flush=True,
            )
        results_df, grouped_row = evaluate_hidden_size_cached(
            data_bundle=data_bundle,
            base_config=base_config,
            n_hidden=clamped_hidden,
            seeds=seeds,
            artifact_root=artifact_root,
            run_prefix=run_prefix,
            results_path=results_path,
            search_stage=search_stage,
            search_order=search_order,
            verbose=verbose,
        )
        mean_val_loss = float(grouped_row["mean_val_loss"])
        is_acceptable = mean_val_loss <= threshold_limit
        grouped_row = {
            **grouped_row,
            "n_hidden": clamped_hidden,
            "is_acceptable": is_acceptable,
        }
        if verbose:
            print(
                f"[hidden] summary n_hidden={clamped_hidden} stage={search_stage} "
                f"mean_val={mean_val_loss:.6e} acceptable={is_acceptable}",
                flush=True,
            )
        search_rows.append(
            {
                "search_order": search_order,
                "search_stage": search_stage,
                "n_hidden": clamped_hidden,
                "mean_val_loss": mean_val_loss,
                "threshold_limit": threshold_limit,
                "is_acceptable": is_acceptable,
            }
        )
        evaluated_hidden.add(clamped_hidden)
        pd.DataFrame(search_rows).to_csv(trace_path, index=False)
        search_order += 1
        return grouped_row, is_acceptable

    start_row, start_ok = evaluate(start_hidden, "initial")

    if start_ok:
        low = minimum_hidden - 1
        high = start_hidden
    else:
        low = start_hidden
        high = None
        probe = start_hidden
        while probe < maximum_hidden:
            next_probe = min(maximum_hidden, max(probe + 1, probe * 2))
            probe_row, probe_ok = evaluate(next_probe, "expand_up")
            if probe_ok:
                high = int(probe_row["n_hidden"])
                break
            low = next_probe
            probe = next_probe

    if high is not None:
        while high - low > 1:
            mid = (low + high) // 2
            mid_row, mid_ok = evaluate(mid, "bisect")
            if mid_ok:
                high = int(mid_row["n_hidden"])
            else:
                low = int(mid_row["n_hidden"])

    results_df = load_hidden_threshold_cache(results_path=results_path)
    evaluated_hidden = set(int(value) for value in results_df["n_hidden"].dropna().unique().tolist())
    threshold_summary = hidden_threshold_from_results(
        results_df=results_df,
        tolerance_ratio=tolerance_ratio,
        reference_loss=reference_loss,
        require_plateau=False,
    )
    if high is None and threshold_summary["minimum_hidden"] is None:
        threshold_summary["selected_hidden"] = threshold_summary["best_hidden"]

    figure_path, threshold_summary = plot_hidden_threshold(
        results_df=results_df,
        save_path=directories["figures"] / f"{run_prefix}_threshold.png",
        tolerance_ratio=tolerance_ratio,
        reference_loss=reference_loss,
        require_plateau=False,
    )
    grouped_results_path = directories["final"] / f"{run_prefix}_grouped_results.csv"
    threshold_summary["grouped_results"].to_csv(grouped_results_path, index=False)
    threshold_json_path = directories["final"] / f"{run_prefix}_threshold.json"
    write_json(
        threshold_json_path,
        {
            "best_loss": threshold_summary["best_loss"],
            "best_hidden": threshold_summary["best_hidden"],
            "threshold_limit": threshold_summary["threshold_limit"],
            "minimum_hidden": threshold_summary["minimum_hidden"],
            "selected_hidden": threshold_summary["selected_hidden"],
            "threshold_found": threshold_summary["threshold_found"],
            "acceptable_hidden": threshold_summary["acceptable_hidden"],
            "reference_loss": threshold_summary["reference_loss"],
            "start_hidden": start_hidden,
            "minimum_hidden_bound": minimum_hidden,
            "maximum_hidden_bound": maximum_hidden,
            "evaluated_hidden": sorted(evaluated_hidden),
            "trace_path": str(trace_path),
            "grouped_results_path": str(grouped_results_path),
        },
    )
    return {
        "results_df": results_df,
        "results_path": results_path,
        "trace_path": trace_path,
        "figure_path": figure_path,
        "grouped_results_path": grouped_results_path,
        "threshold_summary": threshold_summary,
        "threshold_json_path": threshold_json_path,
    }


def run_hidden_threshold_adaptive_progressive(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    seeds: list[int],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_prefix: str = "hidden_threshold",
    start_hidden: int = 4,
    minimum_hidden: int = 1,
    maximum_hidden: int = 48,
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    directories = ensure_artifact_tree(artifact_root)
    stage_rows: list[dict[str, Any]] = []
    stage_paths: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for stage_index in range(len(seeds)):
        active_seeds = seeds[: stage_index + 1]
        if verbose:
            print(
                f"[hidden] progressive stage={stage_index + 1}/{len(seeds)} "
                f"active_seeds={active_seeds}",
                flush=True,
            )
        final_result = run_hidden_threshold_adaptive(
            data_bundle=data_bundle,
            base_config=base_config,
            seeds=active_seeds,
            artifact_root=artifact_root,
            run_prefix=run_prefix,
            start_hidden=start_hidden,
            minimum_hidden=minimum_hidden,
            maximum_hidden=maximum_hidden,
            tolerance_ratio=tolerance_ratio,
            reference_loss=reference_loss,
            verbose=verbose,
        )

        stage_tag = f"stage_{stage_index + 1:02d}"
        stage_figure_path, _ = plot_hidden_threshold(
            results_df=final_result["results_df"],
            save_path=directories["figures"] / f"{run_prefix}_{stage_tag}_threshold.png",
            tolerance_ratio=tolerance_ratio,
            reference_loss=reference_loss,
            require_plateau=False,
        )
        stage_grouped_path = directories["final"] / f"{run_prefix}_{stage_tag}_grouped_results.csv"
        final_result["threshold_summary"]["grouped_results"].to_csv(stage_grouped_path, index=False)
        stage_json_path = directories["final"] / f"{run_prefix}_{stage_tag}_threshold.json"
        write_json(
            stage_json_path,
            {
                "stage_index": stage_index + 1,
                "active_seeds": active_seeds,
                "selected_hidden": final_result["threshold_summary"]["selected_hidden"],
                "minimum_hidden": final_result["threshold_summary"]["minimum_hidden"],
                "best_hidden": final_result["threshold_summary"]["best_hidden"],
                "threshold_limit": final_result["threshold_summary"]["threshold_limit"],
                "reference_loss": final_result["threshold_summary"]["reference_loss"],
                "threshold_found": final_result["threshold_summary"]["threshold_found"],
                "acceptable_hidden": final_result["threshold_summary"]["acceptable_hidden"],
                "figure_path": str(stage_figure_path),
                "grouped_results_path": str(stage_grouped_path),
                "results_path": str(final_result["results_path"]),
                "trace_path": str(final_result["trace_path"]),
            },
        )

        stage_rows.append(
            {
                "stage_index": stage_index + 1,
                "n_active_seeds": len(active_seeds),
                "active_seeds": ",".join(str(seed) for seed in active_seeds),
                "selected_hidden": final_result["threshold_summary"]["selected_hidden"],
                "minimum_hidden": final_result["threshold_summary"]["minimum_hidden"],
                "best_hidden": final_result["threshold_summary"]["best_hidden"],
                "threshold_limit": final_result["threshold_summary"]["threshold_limit"],
                "threshold_found": final_result["threshold_summary"]["threshold_found"],
            }
        )
        stage_paths.append(
            {
                "stage_index": stage_index + 1,
                "figure_path": stage_figure_path,
                "grouped_results_path": stage_grouped_path,
                "threshold_json_path": stage_json_path,
            }
        )
        if verbose:
            print(
                f"[hidden] stage_complete stage={stage_index + 1}/{len(seeds)} "
                f"selected_hidden={final_result['threshold_summary']['selected_hidden']} "
                f"threshold_found={final_result['threshold_summary']['threshold_found']}",
                flush=True,
            )

    if final_result is None:
        raise ValueError("At least one seed is required for the progressive hidden-threshold search.")

    stage_summary_path = directories["final"] / f"{run_prefix}_seed_progression.csv"
    pd.DataFrame(stage_rows).to_csv(stage_summary_path, index=False)
    return {
        **final_result,
        "stage_rows": stage_rows,
        "stage_paths": stage_paths,
        "stage_summary_path": stage_summary_path,
    }


def run_hidden_threshold_grid(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    hidden_grid: list[int],
    seeds: list[int],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_prefix: str = "hidden_threshold_grid",
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Cached grid sweep used when the hidden sizes are fixed in advance."""
    if not hidden_grid:
        raise ValueError("hidden_grid must contain at least one hidden size.")

    directories = ensure_artifact_tree(artifact_root)
    results_path = directories["final"] / f"{run_prefix}_results.csv"
    trace_path = directories["final"] / f"{run_prefix}_trace.csv"

    ordered_hidden = [int(value) for value in hidden_grid]
    threshold_limit = (
        float(reference_loss) * (1.0 + tolerance_ratio)
        if reference_loss is not None
        else np.nan
    )
    trace_rows: list[dict[str, Any]] = []

    for search_order, n_hidden in enumerate(ordered_hidden):
        if verbose:
            print(
                f"[hidden] evaluate n_hidden={n_hidden} stage=grid "
                f"seeds={seeds} order={search_order}",
                flush=True,
            )
        results_df, grouped_row = evaluate_hidden_size_cached(
            data_bundle=data_bundle,
            base_config=base_config,
            n_hidden=n_hidden,
            seeds=seeds,
            artifact_root=artifact_root,
            run_prefix=run_prefix,
            results_path=results_path,
            search_stage="grid",
            search_order=search_order,
            verbose=verbose,
        )
        mean_val_loss = float(grouped_row["mean_val_loss"])
        is_acceptable = (
            bool(mean_val_loss <= threshold_limit)
            if reference_loss is not None
            else np.nan
        )
        trace_rows.append(
            {
                "search_order": search_order,
                "search_stage": "grid",
                "n_hidden": int(n_hidden),
                "mean_val_loss": mean_val_loss,
                "threshold_limit": threshold_limit,
                "is_acceptable": is_acceptable,
            }
        )
        pd.DataFrame(trace_rows).to_csv(trace_path, index=False)
        if verbose:
            acceptable_text = (
                f" acceptable={is_acceptable}"
                if reference_loss is not None
                else ""
            )
            print(
                f"[hidden] summary n_hidden={n_hidden} stage=grid "
                f"mean_val={mean_val_loss:.6e}{acceptable_text}",
                flush=True,
            )

    results_df = load_hidden_threshold_cache(results_path=results_path)
    figure_path, threshold_summary = plot_hidden_threshold(
        results_df=results_df,
        save_path=directories["figures"] / f"{run_prefix}_threshold.png",
        tolerance_ratio=tolerance_ratio,
        reference_loss=reference_loss,
        require_plateau=False if reference_loss is not None else None,
    )
    grouped_results_path = directories["final"] / f"{run_prefix}_grouped_results.csv"
    threshold_summary["grouped_results"].to_csv(grouped_results_path, index=False)
    threshold_json_path = directories["final"] / f"{run_prefix}_threshold.json"
    write_json(
        threshold_json_path,
        {
            "best_loss": threshold_summary["best_loss"],
            "best_hidden": threshold_summary["best_hidden"],
            "threshold_limit": threshold_summary["threshold_limit"],
            "minimum_hidden": threshold_summary["minimum_hidden"],
            "selected_hidden": threshold_summary["selected_hidden"],
            "threshold_found": threshold_summary["threshold_found"],
            "acceptable_hidden": threshold_summary["acceptable_hidden"],
            "reference_loss": threshold_summary["reference_loss"],
            "hidden_grid": ordered_hidden,
            "grouped_results_path": str(grouped_results_path),
            "trace_path": str(trace_path),
        },
    )
    return {
        "results_df": results_df,
        "results_path": results_path,
        "trace_path": trace_path,
        "figure_path": figure_path,
        "grouped_results_path": grouped_results_path,
        "threshold_summary": threshold_summary,
        "threshold_json_path": threshold_json_path,
    }


def run_hidden_threshold_grid_progressive(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    hidden_grid: list[int],
    seeds: list[int],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_prefix: str = "hidden_threshold_grid",
    tolerance_ratio: float = 0.05,
    reference_loss: float | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    directories = ensure_artifact_tree(artifact_root)
    stage_rows: list[dict[str, Any]] = []
    stage_paths: list[dict[str, Any]] = []
    final_result: dict[str, Any] | None = None

    for stage_index in range(len(seeds)):
        active_seeds = seeds[: stage_index + 1]
        if verbose:
            print(
                f"[hidden] progressive stage={stage_index + 1}/{len(seeds)} "
                f"active_seeds={active_seeds} hidden_grid={hidden_grid}",
                flush=True,
            )
        final_result = run_hidden_threshold_grid(
            data_bundle=data_bundle,
            base_config=base_config,
            hidden_grid=hidden_grid,
            seeds=active_seeds,
            artifact_root=artifact_root,
            run_prefix=run_prefix,
            tolerance_ratio=tolerance_ratio,
            reference_loss=reference_loss,
            verbose=verbose,
        )

        stage_tag = f"stage_{stage_index + 1:02d}"
        stage_figure_path, _ = plot_hidden_threshold(
            results_df=final_result["results_df"],
            save_path=directories["figures"] / f"{run_prefix}_{stage_tag}_threshold.png",
            tolerance_ratio=tolerance_ratio,
            reference_loss=reference_loss,
            require_plateau=False if reference_loss is not None else None,
        )
        stage_grouped_path = directories["final"] / f"{run_prefix}_{stage_tag}_grouped_results.csv"
        final_result["threshold_summary"]["grouped_results"].to_csv(stage_grouped_path, index=False)
        stage_json_path = directories["final"] / f"{run_prefix}_{stage_tag}_threshold.json"
        write_json(
            stage_json_path,
            {
                "stage_index": stage_index + 1,
                "active_seeds": active_seeds,
                "hidden_grid": [int(value) for value in hidden_grid],
                "selected_hidden": final_result["threshold_summary"]["selected_hidden"],
                "minimum_hidden": final_result["threshold_summary"]["minimum_hidden"],
                "best_hidden": final_result["threshold_summary"]["best_hidden"],
                "threshold_limit": final_result["threshold_summary"]["threshold_limit"],
                "reference_loss": final_result["threshold_summary"]["reference_loss"],
                "threshold_found": final_result["threshold_summary"]["threshold_found"],
                "acceptable_hidden": final_result["threshold_summary"]["acceptable_hidden"],
                "figure_path": str(stage_figure_path),
                "grouped_results_path": str(stage_grouped_path),
                "results_path": str(final_result["results_path"]),
                "trace_path": str(final_result["trace_path"]),
            },
        )
        stage_rows.append(
            {
                "stage_index": stage_index + 1,
                "n_active_seeds": len(active_seeds),
                "active_seeds": ",".join(str(seed) for seed in active_seeds),
                "selected_hidden": final_result["threshold_summary"]["selected_hidden"],
                "minimum_hidden": final_result["threshold_summary"]["minimum_hidden"],
                "best_hidden": final_result["threshold_summary"]["best_hidden"],
                "threshold_limit": final_result["threshold_summary"]["threshold_limit"],
                "threshold_found": final_result["threshold_summary"]["threshold_found"],
            }
        )
        stage_paths.append(
            {
                "stage_index": stage_index + 1,
                "figure_path": stage_figure_path,
                "grouped_results_path": stage_grouped_path,
                "threshold_json_path": stage_json_path,
            }
        )
        if verbose:
            print(
                f"[hidden] stage_complete stage={stage_index + 1}/{len(seeds)} "
                f"selected_hidden={final_result['threshold_summary']['selected_hidden']} "
                f"threshold_found={final_result['threshold_summary']['threshold_found']}",
                flush=True,
            )

    if final_result is None:
        raise ValueError("At least one seed is required for the progressive hidden-grid search.")

    stage_summary_path = directories["final"] / f"{run_prefix}_seed_progression.csv"
    pd.DataFrame(stage_rows).to_csv(stage_summary_path, index=False)
    return {
        **final_result,
        "stage_rows": stage_rows,
        "stage_paths": stage_paths,
        "stage_summary_path": stage_summary_path,
    }


def retrain_final_model(
    data_bundle: dict[str, Any],
    base_config: ExperimentConfig,
    n_hidden: int,
    fixed_epochs: int,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    run_name: str = "final_model",
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config.N_HIDDEN = n_hidden
    config.MODEL_TAG = run_name
    return train_model(
        data_bundle=data_bundle,
        config=config,
        artifact_root=artifact_root,
        run_name=run_name,
        train_on_train_plus_val=True,
        fixed_epochs=fixed_epochs,
    )


def plot_residual_analysis(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    output_normalizer: MinMaxNormalizer,
    save_path: str | Path,
) -> Path:
    residual_arrays = make_residual_arrays(y_true=y_true, y_pred=y_pred, output_normalizer=output_normalizer)
    y_true_raw = residual_arrays["y_true_raw"]
    y_pred_raw = residual_arrays["y_pred_raw"]
    residual_raw = residual_arrays["residual_raw"]

    mean_abs_residual_over_time = np.mean(np.abs(residual_raw), axis=0)
    flattened_truth = y_true_raw.reshape(-1)
    flattened_pred = y_pred_raw.reshape(-1)
    flattened_residual = residual_raw.reshape(-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].hist(flattened_residual, bins=60, color="#d62728", alpha=0.85)
    axes[0, 0].set_title("Residual histogram")
    axes[0, 1].scatter(flattened_pred, flattened_residual, s=5, alpha=0.15, color="#1f77b4")
    axes[0, 1].set_title("Residual vs predicted")
    axes[0, 1].set_xlabel("predicted stress")
    axes[0, 1].set_ylabel("residual")
    axes[1, 0].plot(mean_abs_residual_over_time, color="#2ca02c", linewidth=1.5)
    axes[1, 0].set_title("Mean absolute residual over time")
    axes[1, 0].set_xlabel("time index")
    axes[1, 0].set_ylabel("mean |residual|")

    if HAS_STATSMODELS:
        sample_index = int(np.argmax(np.mean(np.abs(residual_raw), axis=1)))
        axes[1, 1].stem(acf(residual_raw[sample_index], nlags=min(60, residual_raw.shape[1] - 1)), basefmt=" ")
        axes[1, 1].set_title(f"Residual ACF (sample {sample_index})")
    else:
        axes[1, 1].text(0.5, 0.5, "statsmodels unavailable", ha="center", va="center")
        axes[1, 1].set_title("Residual ACF")
    for axis in axes.reshape(-1):
        axis.grid(alpha=0.3)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_unseen_predictions(
    unseen_cases: dict[str, np.ndarray],
    unseen_predictions: dict[str, np.ndarray],
    save_path: str | Path,
) -> Path:
    n_cases = len(unseen_cases)
    fig, axes = plt.subplots(n_cases, 2, figsize=(12, 3.5 * n_cases))
    axes = np.atleast_2d(axes)
    time_axis = np.linspace(0.0, 1.0, next(iter(unseen_cases.values())).shape[0])
    for row_index, (case_name, strain_history) in enumerate(unseen_cases.items()):
        stress_history = unseen_predictions[case_name]
        axes[row_index, 0].plot(time_axis, strain_history, color="#1f77b4")
        axes[row_index, 0].set_title(f"{case_name}: strain")
        axes[row_index, 0].set_ylabel("strain")
        axes[row_index, 0].grid(alpha=0.3)

        axes[row_index, 1].plot(time_axis, stress_history, color="#d62728")
        axes[row_index, 1].set_title(f"{case_name}: predicted stress")
        axes[row_index, 1].set_ylabel("stress")
        axes[row_index, 1].grid(alpha=0.3)

    axes[-1, 0].set_xlabel("time")
    axes[-1, 1].set_xlabel("time")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return save_path


def predict_unseen_loads(
    checkpoint_path: str | Path,
    input_normalizer: MinMaxNormalizer,
    output_normalizer: MinMaxNormalizer,
    num_points: int = 1001,
) -> dict[str, np.ndarray]:
    device = select_device()
    net, checkpoint = load_checkpoint(checkpoint_path, device=device)
    initial_stress_model = InitialStressModel.from_dict(checkpoint["initial_stress_model"])
    unseen_cases = generate_unseen_load_cases(num_points=num_points)
    unseen_predictions: dict[str, np.ndarray] = {}
    dt = 1.0 / (num_points - 1)

    with torch.no_grad():
        for case_name, strain_history in unseen_cases.items():
            normalized_strain = input_normalizer.transform(strain_history[None, :]).astype(np.float32)
            x = torch.from_numpy(normalized_strain).to(device)
            y_pred = rollout_sequence(net=net, x=x, dt=dt, initial_stress_model=initial_stress_model, y_true0=None)
            y_pred_raw = to_numpy(output_normalizer.inverse_transform(y_pred))[0]
            unseen_predictions[case_name] = y_pred_raw

    return unseen_cases, unseen_predictions


def run_inference_and_testing(
    checkpoint_path: str | Path,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Evaluate a saved checkpoint on the test split and on hand-crafted unseen load cases."""
    data_bundle = prepare_data(artifact_root=artifact_root, split_seed=split_seed)
    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=artifact_root,
        run_name="test_set",
        y_true0=data_bundle["y_test"][:, 0],
    )

    directories = ensure_artifact_tree(artifact_root)
    residual_plot_path = plot_residual_analysis(
        y_true=torch.from_numpy(np.load(evaluation["prediction_path"])["y_true"]),
        y_pred=torch.from_numpy(np.load(evaluation["prediction_path"])["y_pred"]),
        output_normalizer=data_bundle["output_normalizer"],
        save_path=directories["figures"] / "04_residual_analysis.png",
    )

    unseen_cases, unseen_predictions = predict_unseen_loads(
        checkpoint_path=checkpoint_path,
        input_normalizer=data_bundle["input_normalizer"],
        output_normalizer=data_bundle["output_normalizer"],
        num_points=data_bundle["inputsize"],
    )
    unseen_plot_path = plot_unseen_predictions(
        unseen_cases=unseen_cases,
        unseen_predictions=unseen_predictions,
        save_path=directories["figures"] / "04_unseen_predictions.png",
    )
    unseen_predictions_path = directories["predictions"] / "unseen_predictions.npz"
    np.savez(unseen_predictions_path, **{f"{key}_strain": value for key, value in unseen_cases.items()}, **{f"{key}_stress": value for key, value in unseen_predictions.items()})

    metrics_path = directories["reports"] / "test_metrics.json"
    write_json(metrics_path, evaluation["metrics"])
    return {
        "metrics": evaluation["metrics"],
        "prediction_path": evaluation["prediction_path"],
        "residual_plot_path": residual_plot_path,
        "unseen_plot_path": unseen_plot_path,
        "unseen_predictions_path": unseen_predictions_path,
        "metrics_path": metrics_path,
    }


def summarize_test_trajectory_examples(
    strain_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    start_index: int = 1,
) -> pd.DataFrame:
    rmse = np.sqrt(np.mean((y_pred_raw[:, start_index:] - y_true_raw[:, start_index:]) ** 2, axis=1))
    mae = np.mean(np.abs(y_pred_raw[:, start_index:] - y_true_raw[:, start_index:]), axis=1)
    order = np.argsort(rmse)
    selections = {
        "best": int(order[0]),
        "median": int(order[len(order) // 2]),
        "worst": int(order[-1]),
    }
    rows: list[dict[str, Any]] = []
    for label, index in selections.items():
        rows.append(
            {
                "selection": label,
                "sample_index": index,
                "rmse": float(rmse[index]),
                "mae": float(mae[index]),
                "max_abs_stress_error": float(np.max(np.abs(y_pred_raw[index, start_index:] - y_true_raw[index, start_index:]))),
                "strain_min": float(np.min(strain_raw[index])),
                "strain_max": float(np.max(strain_raw[index])),
                "stress_true_min": float(np.min(y_true_raw[index])),
                "stress_true_max": float(np.max(y_true_raw[index])),
            }
        )
    return pd.DataFrame(rows)


def plot_test_stress_strain_examples(
    strain_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    example_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    ordered = ["best", "median", "worst"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for ax, selection in zip(axes, ordered):
        row = example_df.loc[example_df["selection"] == selection].iloc[0]
        idx = int(row["sample_index"])
        ax.plot(strain_raw[idx], y_true_raw[idx], color="black", linewidth=2.0, label="true stress-strain")
        ax.plot(strain_raw[idx], y_pred_raw[idx], color="#d62728", linewidth=1.8, linestyle="--", label="predicted stress-strain")
        ax.set_title(f"{selection.capitalize()} sample\nidx={idx}, RMSE={row['rmse']:.2e}")
        ax.set_xlabel("strain")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("stress")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_test_stress_time_examples(
    strain_raw: np.ndarray,
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    example_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    ordered = ["best", "median", "worst"]
    time_axis = np.linspace(0.0, 1.0, strain_raw.shape[1])
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.0), sharex=True)
    for ax, selection in zip(axes, ordered):
        row = example_df.loc[example_df["selection"] == selection].iloc[0]
        idx = int(row["sample_index"])
        ax.plot(time_axis, y_true_raw[idx], color="black", linewidth=2.0, label="true stress")
        ax.plot(time_axis, y_pred_raw[idx], color="#d62728", linewidth=1.8, linestyle="--", label="predicted stress")
        ax.set_title(f"{selection.capitalize()} sample\nidx={idx}, RMSE={row['rmse']:.2e}")
        ax.set_ylabel("stress")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def hysteresis_loop_area(strain: np.ndarray, stress: np.ndarray) -> float:
    return float(np.abs(np.trapz(stress, strain)))


def plot_hysteresis_checks(
    unseen_cases: dict[str, np.ndarray],
    unseen_predictions: dict[str, np.ndarray],
    save_path: str | Path,
) -> Path:
    cyclic_case_names = [name for name in unseen_cases if ("cycle" in name or "sinusoidal" in name)]
    if not cyclic_case_names:
        raise ValueError("No cyclic unseen load cases were found for the hysteresis check.")

    fig, axes = plt.subplots(1, len(cyclic_case_names), figsize=(6.5 * len(cyclic_case_names), 4.8), squeeze=False)
    axes = axes.ravel()
    for ax, case_name in zip(axes, cyclic_case_names):
        strain = unseen_cases[case_name]
        stress = unseen_predictions[case_name]
        loop_area = hysteresis_loop_area(strain=strain, stress=stress)
        ax.plot(strain, stress, color="#1f77b4", linewidth=2.0)
        ax.set_title(f"{case_name}\nloop area = {loop_area:.3e}")
        ax.set_xlabel("strain")
        ax.set_ylabel("predicted stress")
        ax.grid(alpha=0.3)

    fig.suptitle("Predicted hysteresis under cyclic loading", y=1.02)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_trajectory_and_hysteresis_analysis(
    checkpoint_path: str | Path,
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Generate qualitative best/median/worst plots and cyclic-response hysteresis figures."""
    data_bundle = prepare_data(artifact_root=artifact_root, split_seed=split_seed)
    evaluation = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        x=data_bundle["x_test"],
        y=data_bundle["y_test"],
        output_normalizer=data_bundle["output_normalizer"],
        artifact_root=artifact_root,
        run_name="test_set",
        y_true0=data_bundle["y_test"][:, 0],
    )
    prediction_arrays = np.load(evaluation["prediction_path"])
    strain_raw = to_numpy(data_bundle["input_normalizer"].inverse_transform(data_bundle["x_test"]))
    y_true_raw = prediction_arrays["y_true_raw"]
    y_pred_raw = prediction_arrays["y_pred_raw"]

    example_df = summarize_test_trajectory_examples(
        strain_raw=strain_raw,
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
    )
    directories = ensure_artifact_tree(artifact_root)
    example_summary_path = directories["reports"] / "05_test_trajectory_examples.csv"
    example_df.to_csv(example_summary_path, index=False)

    stress_strain_plot_path = plot_test_stress_strain_examples(
        strain_raw=strain_raw,
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        example_df=example_df,
        save_path=directories["figures"] / "05_test_stress_strain_examples.png",
    )
    stress_time_plot_path = plot_test_stress_time_examples(
        strain_raw=strain_raw,
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        example_df=example_df,
        save_path=directories["figures"] / "05_test_stress_time_examples.png",
    )

    unseen_cases, unseen_predictions = predict_unseen_loads(
        checkpoint_path=checkpoint_path,
        input_normalizer=data_bundle["input_normalizer"],
        output_normalizer=data_bundle["output_normalizer"],
        num_points=data_bundle["inputsize"],
    )
    hysteresis_plot_path = plot_hysteresis_checks(
        unseen_cases=unseen_cases,
        unseen_predictions=unseen_predictions,
        save_path=directories["figures"] / "05_hysteresis_checks.png",
    )

    return {
        "metrics": evaluation["metrics"],
        "example_df": example_df,
        "example_summary_path": example_summary_path,
        "stress_strain_plot_path": stress_strain_plot_path,
        "stress_time_plot_path": stress_time_plot_path,
        "hysteresis_plot_path": hysteresis_plot_path,
        "prediction_path": evaluation["prediction_path"],
    }


LOADING_CLASS_ORDER = [
    "monotonic_slow",
    "monotonic_fast",
    "hold_relaxation",
    "cyclic_or_reversing",
    "mixed_complex",
]

LOADING_CLASS_LABELS = {
    "monotonic_slow": "Monotonic slow",
    "monotonic_fast": "Monotonic fast",
    "hold_relaxation": "Hold / relaxation",
    "cyclic_or_reversing": "Cyclic / reversing",
    "mixed_complex": "Mixed / complex",
}


def make_loading_case_features(strain_raw: np.ndarray) -> pd.DataFrame:
    """Extract simple descriptors that distinguish monotonic, cyclic, and mixed histories."""
    dt = 1.0 / max(strain_raw.shape[1] - 1, 1)
    rows: list[dict[str, Any]] = []
    for sample_index, strain in enumerate(strain_raw):
        diff = np.diff(strain)
        rate = diff / dt
        abs_rate = np.abs(rate)
        max_abs_rate = float(abs_rate.max()) if abs_rate.size else 0.0
        mean_abs_rate = float(abs_rate.mean()) if abs_rate.size else 0.0
        significant_tol = max(1e-8, 0.08 * max_abs_rate)
        hold_tol = max(1e-8, 0.02 * max_abs_rate)
        significant_rate = rate[abs_rate > significant_tol]
        significant_sign = np.sign(significant_rate)
        if significant_sign.size > 1:
            turning_points = int(np.sum(significant_sign[1:] * significant_sign[:-1] < 0.0))
        else:
            turning_points = 0
        hold_fraction = float(np.mean(abs_rate <= hold_tol)) if abs_rate.size else 1.0
        positive_fraction = float(np.mean(rate > significant_tol)) if rate.size else 0.0
        negative_fraction = float(np.mean(rate < -significant_tol)) if rate.size else 0.0
        net_change = float(strain[-1] - strain[0])
        total_variation = float(np.sum(np.abs(diff)))
        variation_ratio = total_variation / (abs(net_change) + 1e-12)
        strain_range = float(np.max(strain) - np.min(strain))
        rows.append(
            {
                "sample_index": int(sample_index),
                "max_abs_rate": max_abs_rate,
                "mean_abs_rate": mean_abs_rate,
                "turning_points": turning_points,
                "hold_fraction": hold_fraction,
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "net_change": net_change,
                "total_variation": total_variation,
                "variation_ratio": variation_ratio,
                "strain_min": float(np.min(strain)),
                "strain_max": float(np.max(strain)),
                "strain_range": strain_range,
            }
        )
    return pd.DataFrame(rows)


def assign_loading_classes(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Map the raw trajectory features into a small, report-friendly set of loading classes."""
    classified_df = feature_df.copy()
    base_label: list[str] = []
    for row in classified_df.itertuples(index=False):
        is_reversing = row.turning_points >= 3 or (
            row.positive_fraction > 0.05 and row.negative_fraction > 0.05 and row.variation_ratio > 1.35
        )
        is_hold = row.hold_fraction >= 0.12 and row.variation_ratio <= 1.45
        is_monotonic = row.turning_points <= 1 and row.variation_ratio <= 1.12
        if is_reversing:
            base_label.append("cyclic_or_reversing")
        elif is_hold:
            base_label.append("hold_relaxation")
        elif is_monotonic:
            base_label.append("monotonic")
        else:
            base_label.append("mixed_complex")
    classified_df["base_label"] = base_label

    monotonic_mask = classified_df["base_label"] == "monotonic"
    if monotonic_mask.any():
        rate_threshold = float(classified_df.loc[monotonic_mask, "max_abs_rate"].median())
    else:
        rate_threshold = float(classified_df["max_abs_rate"].median())

    loading_class: list[str] = []
    for row in classified_df.itertuples(index=False):
        if row.base_label == "monotonic":
            loading_class.append("monotonic_fast" if row.max_abs_rate >= rate_threshold else "monotonic_slow")
        else:
            loading_class.append(str(row.base_label))
    classified_df["loading_class"] = loading_class
    classified_df["loading_class_label"] = classified_df["loading_class"].map(LOADING_CLASS_LABELS)
    return classified_df


def summarize_sample_prediction_metrics(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    start_index: int = 1,
) -> pd.DataFrame:
    """Compute per-sample errors so model performance can be grouped by loading class."""
    truth = y_true_raw[:, start_index:]
    pred = y_pred_raw[:, start_index:]
    residual = pred - truth
    rmse = np.sqrt(np.mean(residual**2, axis=1))
    mae = np.mean(np.abs(residual), axis=1)
    max_abs_error = np.max(np.abs(residual), axis=1)
    truth_l2 = np.linalg.norm(truth, axis=1)
    residual_l2 = np.linalg.norm(residual, axis=1)
    relative_l2 = np.divide(residual_l2, truth_l2, out=np.zeros_like(residual_l2), where=truth_l2 > 1e-12)
    return pd.DataFrame(
        {
            "sample_index": np.arange(y_true_raw.shape[0], dtype=int),
            "sample_rmse": rmse.astype(float),
            "sample_mae": mae.astype(float),
            "sample_max_abs_error": max_abs_error.astype(float),
            "sample_relative_l2": relative_l2.astype(float),
        }
    )


def plot_loading_case_counts(class_df: pd.DataFrame, save_path: str | Path) -> Path:
    class_counts = (
        class_df.groupby(["loading_class", "loading_class_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    present_order = [name for name in LOADING_CLASS_ORDER if name in class_counts["loading_class"].tolist()]
    class_counts["order"] = class_counts["loading_class"].map({name: i for i, name in enumerate(present_order)})
    class_counts = class_counts.sort_values("order")

    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.bar(class_counts["loading_class_label"], class_counts["count"], color="#4e79a7", alpha=0.9)
    ax.set_title("Test-set loading-case counts")
    ax.set_ylabel("Number of samples")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_loading_case_representatives(
    strain_raw: np.ndarray,
    class_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    present_order = [name for name in LOADING_CLASS_ORDER if name in class_df["loading_class"].tolist()]
    time_axis = np.linspace(0.0, 1.0, strain_raw.shape[1])
    fig, axes = plt.subplots(len(present_order), 1, figsize=(10.0, 2.5 * max(len(present_order), 1)), sharex=True)
    if len(present_order) == 1:
        axes = [axes]
    for ax, loading_class in zip(axes, present_order):
        subset = class_df.loc[class_df["loading_class"] == loading_class].copy()
        median_rate = float(subset["max_abs_rate"].median())
        representative_row = subset.iloc[int(np.argmin(np.abs(subset["max_abs_rate"].to_numpy() - median_rate)))]
        sample_index = int(representative_row["sample_index"])
        ax.plot(time_axis, strain_raw[sample_index], color="#1f77b4", linewidth=2.0)
        ax.set_title(
            f"{LOADING_CLASS_LABELS[loading_class]} "
            f"(n={len(subset)}, idx={sample_index}, max|rate|={representative_row['max_abs_rate']:.2e})"
        )
        ax.set_ylabel("strain")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_loading_case_relative_l2(
    grouped_metrics_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    present_order = [name for name in LOADING_CLASS_ORDER if name in grouped_metrics_df["loading_class"].tolist()]
    class_labels = [LOADING_CLASS_LABELS[name] for name in present_order]
    model_labels = list(grouped_metrics_df["model_label"].drop_duplicates())
    model_colors: dict[str, str] = {}
    for model_label in model_labels:
        color_values = grouped_metrics_df.loc[grouped_metrics_df["model_label"] == model_label, "model_color"].dropna().unique().tolist()
        model_colors[model_label] = color_values[0] if color_values else "#1f77b4"

    x_positions = np.arange(len(present_order), dtype=float)
    width = 0.8 / max(len(model_labels), 1)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for model_idx, model_label in enumerate(model_labels):
        subset = (
            grouped_metrics_df.loc[grouped_metrics_df["model_label"] == model_label]
            .set_index("loading_class")
            .reindex(present_order)
        )
        ax.bar(
            x_positions + (model_idx - (len(model_labels) - 1) / 2.0) * width,
            subset["mean_sample_relative_l2"],
            width=width,
            color=model_colors[model_label],
            alpha=0.88,
            label=model_label,
        )
    ax.set_title("Mean test relative L2 by loading class")
    ax.set_xlabel("loading class")
    ax.set_ylabel("Mean sample relative L2")
    ax.set_yscale("log")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(class_labels, rotation=20)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_loading_case_relative_l2_heatmap(
    grouped_metrics_df: pd.DataFrame,
    save_path: str | Path,
) -> Path:
    present_order = [name for name in LOADING_CLASS_ORDER if name in grouped_metrics_df["loading_class"].tolist()]
    pivot = (
        grouped_metrics_df.pivot(index="model_label", columns="loading_class", values="mean_sample_relative_l2")
        .reindex(columns=present_order)
    )
    fig, ax = plt.subplots(figsize=(9.0, 1.4 + 0.8 * max(len(pivot.index), 1)))
    image = ax.imshow(np.log10(pivot.to_numpy(dtype=float)), aspect="auto", cmap="viridis")
    ax.set_title("log10 mean test relative L2 by model and loading class")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([LOADING_CLASS_LABELS[name] for name in pivot.columns], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index))
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("log10(relative L2)")
    for row_index, model_label in enumerate(pivot.index):
        for col_index, loading_class in enumerate(pivot.columns):
            value = float(pivot.loc[model_label, loading_class])
            ax.text(col_index, row_index, f"{value:.2e}", ha="center", va="center", color="white", fontsize=8)
    fig.tight_layout()
    save_path = Path(save_path)
    ensure_directory(save_path.parent)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_default_loading_case_model_specs(
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
) -> list[dict[str, Any]]:
    """Assemble a compact comparison set from the already-saved checkpoints."""
    artifact_root = Path(artifact_root)
    model_specs: list[dict[str, Any]] = []

    best_param_paths = sorted((artifact_root / "optuna").glob("best_params_*.json"))
    if best_param_paths:
        core_type, payload, _ = pick_best_family(best_param_paths)
        model_specs.append(
            {
                "label": f"best_{core_type}",
                "checkpoint_path": str(payload["best_user_attrs"]["checkpoint_path"]),
                "color": "#2ca02c",
            }
        )

    paper_comparison_path = artifact_root / "reports" / "08_paper_rno_h0_comparison.csv"
    if paper_comparison_path.exists():
        comparison_df = pd.read_csv(paper_comparison_path)
        for row in comparison_df.itertuples(index=False):
            variant = str(getattr(row, "variant"))
            if variant == "paper_rno_no_rate":
                label = "paper_h0_no_rate"
                color = "#1f77b4"
            elif variant == "paper_rno_with_rate":
                label = "paper_h0_with_rate"
                color = "#d62728"
            else:
                label = variant
                color = "#9467bd"
            model_specs.append(
                {
                    "label": label,
                    "checkpoint_path": str(getattr(row, "checkpoint_path")),
                    "color": color,
                }
            )
    return model_specs


def run_loading_case_analysis(
    model_specs: list[dict[str, Any]],
    artifact_root: str | Path = DEFAULT_ARTIFACT_ROOT,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> dict[str, Any]:
    """Classify the test set once, evaluate selected models, and save per-class summaries."""
    directories = ensure_artifact_tree(artifact_root)
    data_bundle = prepare_data(artifact_root=artifact_root, split_seed=split_seed)
    strain_raw = to_numpy(data_bundle["input_normalizer"].inverse_transform(data_bundle["x_test"]))

    class_df = assign_loading_classes(make_loading_case_features(strain_raw=strain_raw))
    class_counts_df = (
        class_df.groupby(["loading_class", "loading_class_label"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    sample_metric_frames: list[pd.DataFrame] = []
    normalized_model_specs: list[dict[str, Any]] = []
    for model_index, spec in enumerate(model_specs):
        checkpoint_path = Path(spec["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        label = str(spec["label"])
        color = str(spec.get("color", ["#2ca02c", "#1f77b4", "#d62728", "#9467bd"][model_index % 4]))
        run_stub = "".join(ch if ch.isalnum() else "_" for ch in label.lower()).strip("_")
        evaluation = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            x=data_bundle["x_test"],
            y=data_bundle["y_test"],
            output_normalizer=data_bundle["output_normalizer"],
            artifact_root=artifact_root,
            run_name=f"10_{run_stub}",
            y_true0=data_bundle["y_test"][:, 0],
        )
        prediction_arrays = np.load(evaluation["prediction_path"])
        sample_metrics_df = summarize_sample_prediction_metrics(
            y_true_raw=prediction_arrays["y_true_raw"],
            y_pred_raw=prediction_arrays["y_pred_raw"],
        )
        sample_metrics_df["model_label"] = label
        sample_metrics_df["model_color"] = color
        sample_metrics_df["checkpoint_path"] = str(checkpoint_path)
        sample_metric_frames.append(sample_metrics_df.merge(class_df, on="sample_index", how="left"))
        normalized_model_specs.append({"label": label, "checkpoint_path": str(checkpoint_path), "color": color})

    sample_metrics_all = pd.concat(sample_metric_frames, ignore_index=True)
    grouped_metrics_df = (
        sample_metrics_all.groupby(["model_label", "model_color", "loading_class", "loading_class_label"], as_index=False)
        .agg(
            n_samples=("sample_index", "size"),
            mean_sample_relative_l2=("sample_relative_l2", "mean"),
            median_sample_relative_l2=("sample_relative_l2", "median"),
            mean_sample_rmse=("sample_rmse", "mean"),
            mean_sample_mae=("sample_mae", "mean"),
            mean_sample_max_abs_error=("sample_max_abs_error", "mean"),
        )
    )
    grouped_metrics_df["class_order"] = grouped_metrics_df["loading_class"].map({name: idx for idx, name in enumerate(LOADING_CLASS_ORDER)})
    grouped_metrics_df = grouped_metrics_df.sort_values(["class_order", "model_label"]).drop(columns="class_order").reset_index(drop=True)

    model_specs_df = pd.DataFrame(normalized_model_specs)
    model_specs_path = directories["reports"] / "10_loading_case_model_specs.csv"
    class_features_path = directories["reports"] / "10_loading_case_features.csv"
    sample_metrics_path = directories["reports"] / "10_loading_case_sample_metrics.csv"
    grouped_metrics_path = directories["reports"] / "10_loading_case_grouped_metrics.csv"
    class_counts_path = directories["reports"] / "10_loading_case_class_counts.csv"
    model_specs_df.to_csv(model_specs_path, index=False)
    class_df.to_csv(class_features_path, index=False)
    sample_metrics_all.to_csv(sample_metrics_path, index=False)
    grouped_metrics_df.to_csv(grouped_metrics_path, index=False)
    class_counts_df.to_csv(class_counts_path, index=False)

    counts_plot_path = plot_loading_case_counts(
        class_df=class_df,
        save_path=directories["figures"] / "10_loading_case_counts.png",
    )
    representatives_plot_path = plot_loading_case_representatives(
        strain_raw=strain_raw,
        class_df=class_df,
        save_path=directories["figures"] / "10_loading_case_representatives.png",
    )
    relative_l2_plot_path = plot_loading_case_relative_l2(
        grouped_metrics_df=grouped_metrics_df,
        save_path=directories["figures"] / "10_loading_case_relative_l2.png",
    )
    heatmap_plot_path = plot_loading_case_relative_l2_heatmap(
        grouped_metrics_df=grouped_metrics_df,
        save_path=directories["figures"] / "10_loading_case_relative_l2_heatmap.png",
    )

    summary_path = directories["reports"] / "10_loading_case_analysis_summary.json"
    write_json(
        summary_path,
        {
            "model_specs_csv": str(model_specs_path),
            "class_features_csv": str(class_features_path),
            "sample_metrics_csv": str(sample_metrics_path),
            "grouped_metrics_csv": str(grouped_metrics_path),
            "class_counts_csv": str(class_counts_path),
            "counts_plot_path": str(counts_plot_path),
            "representatives_plot_path": str(representatives_plot_path),
            "relative_l2_plot_path": str(relative_l2_plot_path),
            "heatmap_plot_path": str(heatmap_plot_path),
        },
    )

    return {
        "model_specs_df": model_specs_df,
        "class_df": class_df,
        "class_counts_df": class_counts_df,
        "sample_metrics_df": sample_metrics_all,
        "grouped_metrics_df": grouped_metrics_df,
        "model_specs_path": model_specs_path,
        "class_features_path": class_features_path,
        "sample_metrics_path": sample_metrics_path,
        "grouped_metrics_path": grouped_metrics_path,
        "class_counts_path": class_counts_path,
        "counts_plot_path": counts_plot_path,
        "representatives_plot_path": representatives_plot_path,
        "relative_l2_plot_path": relative_l2_plot_path,
        "heatmap_plot_path": heatmap_plot_path,
        "summary_path": summary_path,
    }


def pick_best_family(best_param_paths: list[str | Path]) -> tuple[str, dict[str, Any], Path]:
    best_payload: dict[str, Any] | None = None
    best_path: Path | None = None
    for path in best_param_paths:
        payload = read_json(path)
        if best_payload is None or payload["best_value"] < best_payload["best_value"]:
            best_payload = payload
            best_path = Path(path)
    if best_payload is None or best_path is None:
        raise FileNotFoundError("No Optuna best-parameter files were found.")
    return best_payload["core_type"], best_payload, best_path
