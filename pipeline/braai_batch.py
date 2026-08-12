"""Batch BRAAI inference helpers for 63x63 alert triplets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class BraaiModelBundle:
    """Loaded BRAAI model plus preprocessing settings."""

    model: object
    triplet_size: int = 63
    ref_flip_lr: bool = True


def load_braai_model(model_path: str | Path):
    """Load a Keras BRAAI model with a helpful error if TensorFlow is missing."""
    try:
        import json
        import h5py
        import tensorflow as tf
    except Exception as exc:  # pragma: no cover - import-time dependency guard
        raise RuntimeError("TensorFlow/Keras is required to load BRAAI models") from exc

    model_path = Path(model_path)
    try:
        with h5py.File(model_path, "r") as handle:
            raw_config = handle.attrs.get("model_config")
            if raw_config is None:
                raise ValueError("BRAAI model is missing model_config metadata")
            config_text = raw_config.decode("utf-8") if isinstance(raw_config, bytes) else raw_config
            config = json.loads(config_text)

        if config.get("class_name") != "Sequential":
            raise ValueError(f"Unsupported BRAAI model class: {config.get('class_name')}")

        layers = config["config"]["layers"]
        model = tf.keras.Sequential(name=config["config"].get("name", "braai"))
        built_input = False

        for layer_cfg in layers:
            class_name = layer_cfg.get("class_name")
            cfg = dict(layer_cfg.get("config", {}))

            if class_name == "Conv2D" and not built_input:
                cfg.pop("batch_input_shape", None)
                cfg.pop("batch_shape", None)
                model.add(tf.keras.Input(shape=(63, 63, 3), name="input_layer"))
                built_input = True

            layer_cls = getattr(tf.keras.layers, class_name, None)
            if layer_cls is None:
                raise ValueError(f"Unsupported BRAAI layer class: {class_name}")

            if class_name == "Conv2D":
                layer = layer_cls(
                    filters=cfg["filters"],
                    kernel_size=tuple(cfg["kernel_size"]),
                    strides=tuple(cfg.get("strides", (1, 1))),
                    padding=cfg.get("padding", "valid"),
                    activation=cfg.get("activation"),
                    use_bias=cfg.get("use_bias", True),
                    kernel_initializer=cfg.get("kernel_initializer", "glorot_uniform"),
                    bias_initializer=cfg.get("bias_initializer", "zeros"),
                    kernel_regularizer=cfg.get("kernel_regularizer"),
                    bias_regularizer=cfg.get("bias_regularizer"),
                    activity_regularizer=cfg.get("activity_regularizer"),
                    kernel_constraint=cfg.get("kernel_constraint"),
                    bias_constraint=cfg.get("bias_constraint"),
                    dilation_rate=tuple(cfg.get("dilation_rate", (1, 1))),
                    data_format=cfg.get("data_format", "channels_last"),
                    name=cfg.get("name"),
                    trainable=cfg.get("trainable", True),
                    dtype=cfg.get("dtype", "float32"),
                )
            elif class_name == "MaxPooling2D":
                layer = layer_cls(
                    pool_size=tuple(cfg.get("pool_size", (2, 2))),
                    strides=tuple(cfg.get("strides", cfg.get("pool_size", (2, 2)))),
                    padding=cfg.get("padding", "valid"),
                    data_format=cfg.get("data_format", "channels_last"),
                    name=cfg.get("name"),
                    trainable=cfg.get("trainable", True),
                    dtype=cfg.get("dtype", "float32"),
                )
            elif class_name == "Dropout":
                layer = layer_cls(
                    rate=cfg["rate"],
                    noise_shape=cfg.get("noise_shape"),
                    seed=cfg.get("seed"),
                    name=cfg.get("name"),
                    trainable=cfg.get("trainable", True),
                    dtype=cfg.get("dtype", "float32"),
                )
            elif class_name == "Flatten":
                layer = layer_cls(
                    data_format=cfg.get("data_format", "channels_last"),
                    name=cfg.get("name"),
                    trainable=cfg.get("trainable", True),
                    dtype=cfg.get("dtype", "float32"),
                )
            elif class_name == "Dense":
                layer = layer_cls(
                    units=cfg["units"],
                    activation=cfg.get("activation"),
                    use_bias=cfg.get("use_bias", True),
                    kernel_initializer=cfg.get("kernel_initializer", "glorot_uniform"),
                    bias_initializer=cfg.get("bias_initializer", "zeros"),
                    kernel_regularizer=cfg.get("kernel_regularizer"),
                    bias_regularizer=cfg.get("bias_regularizer"),
                    activity_regularizer=cfg.get("activity_regularizer"),
                    kernel_constraint=cfg.get("kernel_constraint"),
                    bias_constraint=cfg.get("bias_constraint"),
                    name=cfg.get("name"),
                    trainable=cfg.get("trainable", True),
                    dtype=cfg.get("dtype", "float32"),
                )
            else:
                raise ValueError(f"Unsupported BRAAI layer class: {class_name}")

            model.add(layer)

        model.load_weights(str(model_path))
        return model
    except Exception:
        from tensorflow.keras.models import load_model

        return load_model(str(model_path), compile=False)

def _normalize_channel(channel):
    if np.ma.isMaskedArray(channel):
        channel = channel.filled(0.0)

    channel = np.asarray(channel, dtype=np.float32)

    channel = np.nan_to_num(
        channel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    # Replace masked sentinel values
    channel[channel <= -4700] = 0.0

    norm = np.linalg.norm(channel)

    if norm > 0:
        channel /= norm

    return channel

def _ensure_triplet_array(triplet: np.ndarray, triplet_size: int) -> np.ndarray:
    arr = np.asarray(triplet, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"Expected a (H, W, 3) triplet, got {arr.shape}")

    if arr.shape[0] != triplet_size or arr.shape[1] != triplet_size:
        raise ValueError(f"Expected {triplet_size}x{triplet_size} triplets, got {arr.shape[:2]}")

    # Normalize each cutout channel independently
    for i in range(3):
        arr[:, :, i] = _normalize_channel(arr[:, :, i])

    return arr


def predict_braai_batch(triplets: Sequence[np.ndarray], model, *, verbose: int = 0) -> np.ndarray:
    """Run BRAAI inference on a batch of triplets."""
    if not triplets:
        return np.asarray([], dtype=float)

    X = np.stack([np.asarray(triplet, dtype=np.float32) for triplet in triplets], axis=0)
    scores = model.predict(X, verbose=verbose)
    scores = np.asarray(scores).reshape(len(triplets), -1)
    return scores[:, 0].astype(float)


def build_triplet_batch(
    triplets: Iterable[np.ndarray],
    *,
    triplet_size: int = 63,
    ref_flip_lr: bool = True,
) -> list[np.ndarray]:
    """Validate and normalize a batch of triplets before inference."""
    prepared: list[np.ndarray] = []
    for triplet in triplets:
        arr = _ensure_triplet_array(triplet, triplet_size)
        if ref_flip_lr:
            arr = arr.copy()
            arr[:, :, 1] = arr[:, ::-1, 1]
        prepared.append(arr)
    return prepared
