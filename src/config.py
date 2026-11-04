"""Config loader: one YAML file -> one validated, immutable Config object.

Resolution order (later wins):
    1. base YAML
    2. profile block   (profiles.<run.profile>)
    3. dataset block's own `time` overrides (merged into the shared `time`)
    4. CLI overrides   (--set a.b.c=value)
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIT_SECONDS = {"seconds": 1, "minutes": 60, "hours": 3600, "days": 86400}
TimeUnit = Literal["seconds", "minutes", "hours", "days"]


class _Strict(BaseModel):
    """Frozen + extra='forbid', so typos in the YAML fail loudly."""
    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------- run / data
class RunConfig(_Strict):
    profile: str
    seed: int = 42
    base_dir: str = "."
    output_dir: str = "outputs"
    cache_dir: str = "cache"

    @property
    def base_path(self) -> Path:
        return PROJECT_ROOT if self.base_dir == "." else Path(self.base_dir).resolve()

    @property
    def output_path(self) -> Path:
        return self.base_path / self.output_dir

    @property
    def cache_path(self) -> Path:
        return self.base_path / self.cache_dir


class DatasetPaths(_Strict):
    votes: str
    edges: str
    owners: str | None = None


class DatasetColumns(_Strict):
    votes: list[str]
    edges: list[str]


class DatasetConfig(_Strict):
    paths: DatasetPaths
    read_csv_kwargs: dict[str, Any] = Field(default_factory=dict)
    columns: DatasetColumns
    edge_source: str
    edge_target: str
    edge_time_available: bool
    author_rule: Literal["first_event", "owner_file"]
    post_definition: Literal["column", "derived"] = "column"
    time_format: str
    time: dict[str, Any] | None = None

    @model_validator(mode="after")
    def _check(self) -> "DatasetConfig":
        required = ("user", "time") + (("post",) if self.post_definition == "column" else ())
        for col in required:
            if col not in self.columns.votes:
                raise ValueError(f"columns.votes must contain '{col}'")
        for col in (self.edge_source, self.edge_target):
            if col not in self.columns.edges:
                raise ValueError(f"edge column '{col}' not in columns.edges")
        if self.edge_source == self.edge_target:
            raise ValueError("edge_source and edge_target must differ")
        if self.edge_time_available and "created_at" not in self.columns.edges:
            raise ValueError("edge_time_available=true needs 'created_at' in columns.edges")
        if self.author_rule == "owner_file" and not self.paths.owners:
            raise ValueError("author_rule=owner_file needs paths.owners")
        if not re.fullmatch(r"unix_seconds|unix_millis|date:.+", self.time_format):
            raise ValueError("time_format must be unix_seconds, unix_millis or date:<fmt>")
        return self

    def resolve(self, base: Path, key: str) -> Path | None:
        p = getattr(self.paths, key)
        if p is None:
            return None
        pp = Path(p)
        return pp if pp.is_absolute() else base / pp


# --------------------------------------------------------------------- time
class ObsWindow(_Strict):
    mode: Literal["fixed", "fraction_of_duration"]
    value: float = Field(gt=0)
    unit: TimeUnit | None = None

    @model_validator(mode="after")
    def _check(self) -> "ObsWindow":
        if self.mode == "fixed" and self.unit is None:
            raise ValueError("observation_window.mode=fixed needs a unit")
        if self.mode == "fraction_of_duration" and not 0 < self.value < 1:
            raise ValueError("fraction_of_duration needs 0 < value < 1")
        return self

    @property
    def seconds(self) -> float:
        if self.mode != "fixed":
            raise ValueError("window is a fraction; compute it per post")
        return self.value * UNIT_SECONDS[self.unit]


class Horizon(_Strict):
    value: float = Field(gt=0)
    unit: TimeUnit

    @property
    def seconds(self) -> float:
        return self.value * UNIT_SECONDS[self.unit]


class Split(_Strict):
    mode: Literal["quantile"]
    train: float
    val: float
    test: float

    @model_validator(mode="after")
    def _sum(self) -> "Split":
        if abs(self.train + self.val + self.test - 1.0) > 1e-6:
            raise ValueError("split fractions must sum to 1")
        return self


class TimeConfig(_Strict):
    observation_window: ObsWindow
    horizon: Horizon
    split: Split
    purge_straddling_posts: bool = True
    drop_if_horizon_exceeds_data_end: bool = True
    min_observed_events: int = Field(ge=0)

    @model_validator(mode="after")
    def _order(self) -> "TimeConfig":
        w = self.observation_window
        if w.mode == "fixed" and w.seconds >= self.horizon.seconds:
            raise ValueError("observation window must be shorter than the horizon")
        return self


# ---------------------------------------------------------- method sections
class Sampling(_Strict):
    n_posts: int | None = None
    max_users: int | None = None
    keep_time_order: bool = True
    neighborhood_hops: int = 2


class Smoothing(_Strict):
    alpha: float = 1.0
    beta: float = 2.0


class ScaleFactor(_Strict):
    enabled: bool = True
    fit_on: Literal["train", "val"] = "val"


class ExternalArrival(_Strict):
    enabled: bool = False


class FitConfig(_Strict):
    p_estimator: Literal["edge", "user", "global", "backoff"] = "backoff"
    smoothing: Smoothing = Smoothing()
    delay_distribution: Literal["exponential", "weibull"] = "exponential"
    scale_factor: ScaleFactor = ScaleFactor()
    external_arrival: ExternalArrival = ExternalArrival()


class RecencyWeighting(_Strict):
    enabled: bool = False
    half_life: Horizon | None = None


class SimulateConfig(_Strict):
    n_runs: int = Field(gt=0)
    start_active_set: Literal["observed", "author_only"] = "observed"
    use_common_random_numbers: bool = True
    max_cascade_size: int = 200_000
    recency_weighting: RecencyWeighting = RecencyWeighting()


class IMConfig(_Strict):
    k_list: list[int] = [1, 2, 3]
    candidate_hops: int = 2
    candidate_top_n: int = Field(gt=0)
    selection_runs: int = Field(gt=0)
    algorithm: Literal["celf"] = "celf"


class EvalConfig(_Strict):
    metrics: list[Literal["msle", "spearman", "mape"]]
    report_mean_of_k: bool = True


# --------------------------------------------------------------------- ml
class HGTConfig(_Strict):
    hidden_channels: int = 64
    num_heads: int = 4
    num_layers: int = 2


class RGCNConfig(_Strict):
    hidden_channels: int = 64
    num_layers: int = 2


class XGBoostConfig(_Strict):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1


class MLConfig(_Strict):
    epochs: int = 50
    batch_size: int = 32
    lr: float = 0.001
    hgt: HGTConfig = HGTConfig()
    rgcn: RGCNConfig = RGCNConfig()
    xgboost: XGBoostConfig = XGBoostConfig()


# --------------------------------------------------------------------- root
class Config(_Strict):
    run: RunConfig
    dataset: dict[str, str]
    datasets: dict[str, DatasetConfig]
    time: TimeConfig
    sampling: Sampling = Sampling()
    fit: FitConfig
    simulate: SimulateConfig
    im: IMConfig
    eval: EvalConfig
    ml: MLConfig = MLConfig()

    @model_validator(mode="after")
    def _active_exists(self) -> "Config":
        if self.dataset.get("name") not in self.datasets:
            raise ValueError(f"dataset.name must be one of {list(self.datasets)}")
        return self

    @property
    def dataset_name(self) -> str:
        return self.dataset["name"]

    @property
    def active(self) -> DatasetConfig:
        return self.datasets[self.dataset_name]

    def path(self, key: Literal["votes", "edges", "owners"]) -> Path | None:
        return self.active.resolve(self.run.base_path, key)

    def section_hash(self, *sections: str) -> str:
        payload = {s: getattr(self, s).model_dump(mode="json")
                   if hasattr(getattr(self, s), "model_dump") else getattr(self, s)
                   for s in sections}
        payload["dataset_name"] = self.dataset_name
        payload["dataset_block"] = self.active.model_dump(mode="json")
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    def save_resolved(self, out_dir: Path) -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "config.resolved.yaml"
        target.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
        return target


# ------------------------------------------------------------------ loading
def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _apply_override(raw: dict, expr: str) -> None:
    if "=" not in expr:
        raise ValueError(f"override must look like a.b=value, got: {expr}")
    key, val = expr.split("=", 1)
    node = raw
    parts = key.strip().split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = yaml.safe_load(val)


def load_config(config_path: str | Path = "config.yaml",
                overrides: list[str] | None = None) -> Config:
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    raw = yaml.safe_load(path.read_text())

    for expr in overrides or []:
        _apply_override(raw, expr)

    profiles = raw.pop("profiles", {}) or {}
    profile = raw["run"]["profile"]
    if profile not in profiles:
        raise ValueError(f"profile '{profile}' not in profiles: {list(profiles)}")
    raw = _deep_merge(raw, profiles[profile])

    for name, block in raw["datasets"].items():
        ds_time = block.pop("time", None)
        if name == raw["dataset"]["name"] and ds_time:
            raw["time"] = _deep_merge(raw["time"], ds_time)

    for expr in overrides or []:
        _apply_override(raw, expr)

    cfg = Config(**raw)
    log.info("config loaded: dataset=%s profile=%s", cfg.dataset_name, cfg.run.profile)
    return cfg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    c = load_config()
    log.info("base=%s window=%s horizon=%s",
             c.run.base_path, c.time.observation_window, c.time.horizon)