import sys
import types
from importlib.machinery import ModuleSpec
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from alias.util.artifacts import create_run_directory, save_embedding_frame


def test_pseudotime_is_exported_from_alias_evaluation(monkeypatch):
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.__spec__ = ModuleSpec("psutil", loader=None)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    fake_adjust_text = types.ModuleType("adjustText")
    fake_adjust_text.adjust_text = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "adjustText", fake_adjust_text)

    import alias.evaluation as ev

    assert hasattr(ev, "PseudotimeConfig")
    assert hasattr(ev, "pseudotime")


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "alias" / "evaluation" / "pseudotime.py"
)


def load_pseudotime_module(monkeypatch):
    fake_psutil = types.ModuleType("psutil")
    fake_psutil.__spec__ = ModuleSpec("psutil", loader=None)
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    fake_adjust_text = types.ModuleType("adjustText")
    fake_adjust_text.adjust_text = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "adjustText", fake_adjust_text)

    fake_scanpy = types.ModuleType("scanpy")

    def fake_pca(adata, *args, **kwargs):
        n_rows = len(adata.obs.index)
        adata.obsm["X_pca"] = np.column_stack(
            [np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 1.0]
        )

    def fake_neighbors(*args, **kwargs):
        return None

    def fake_umap(adata, *args, **kwargs):
        n_rows = len(adata.obs.index)
        adata.obsm["X_umap"] = np.column_stack(
            [np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 0.5]
        )

    def fake_diffmap(*args, **kwargs):
        return None

    def fake_dpt(adata, copy=False):
        n_rows = len(adata.obs.index)
        adata.obs["dpt_pseudotime"] = np.linspace(0.0, 1.0, n_rows)

    fake_scanpy.pp = types.SimpleNamespace(neighbors=fake_neighbors)
    fake_scanpy.tl = types.SimpleNamespace(
        pca=fake_pca,
        umap=fake_umap,
        diffmap=fake_diffmap,
        dpt=fake_dpt,
    )
    monkeypatch.setitem(sys.modules, "scanpy", fake_scanpy)

    fake_scvi = types.ModuleType("scvi")

    class FakeSCVI:
        setup_calls = []
        init_adatas = []
        train_calls = []

        @staticmethod
        def setup_anndata(adata, **kwargs):
            FakeSCVI.setup_calls.append({"adata": adata, "kwargs": kwargs})

        def __init__(self, adata):
            self.adata = adata
            FakeSCVI.init_adatas.append(adata)

        def train(self, **kwargs):
            FakeSCVI.train_calls.append(kwargs)

        def get_latent_representation(self):
            n_rows = len(self.adata.obs.index)
            return np.column_stack(
                [np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 2.0]
            )

    fake_scvi.model = types.SimpleNamespace(SCVI=FakeSCVI)
    monkeypatch.setitem(sys.modules, "scvi", fake_scvi)

    spec = importlib.util.spec_from_file_location("alias.evaluation.pseudotime", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, "alias.evaluation.pseudotime", module)
    spec.loader.exec_module(module)
    return module


class DummyPlotter:
    def __init__(self, *args, **kwargs):
        self.annotate_centroids = False

    def _touch(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("plot", encoding="utf-8")

    def plot_pseudotime_scatter_time(self, expr_ranks, llm_ranks, time, output_path, title=None):
        self._touch(output_path)

    def plot_pseudotime_scatter_celltypes(self, expr_ranks, llm_ranks, cell_types, output_path, title=None):
        self._touch(output_path)

    def plot_cells(
        self,
        df,
        output_path,
        title,
        annotation_column=None,
        annotate_centroids_df=None,
        continuous_color_column=None,
        time_color_column=None,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        self._touch(output_path)


class FakeAdata:
    def __init__(
        self,
        obs: pd.DataFrame,
        X: np.ndarray,
        var_names: list[str],
        obsm: dict[str, np.ndarray] | None = None,
        layers: dict[str, np.ndarray] | None = None,
    ):
        self.obs = obs
        self.X = X
        self.var_names = pd.Index(var_names)
        self.obsm = dict(obsm or {})
        self.layers = dict(layers or {})
        self.uns = {}
        self.obs_names = self.obs.index

    def copy(self):
        return FakeAdata(
            self.obs.copy(),
            self.X.copy(),
            self.var_names.tolist(),
            {key: value.copy() for key, value in self.obsm.items()},
            {key: value.copy() for key, value in self.layers.items()},
        )

    def __getitem__(self, item):
        if isinstance(item, tuple):
            item = item[0]
        if hasattr(item, "tolist"):
            item = item.tolist()
        if isinstance(item, pd.Index):
            item = item.tolist()
        if item and isinstance(item[0], bool):
            mask = np.asarray(item, dtype=bool)
            obs = self.obs.loc[self.obs.index[mask]].copy()
            X = self.X[mask].copy()
            obsm = {key: value[mask].copy() for key, value in self.obsm.items()}
            layers = {key: value[mask].copy() for key, value in self.layers.items()}
            return FakeAdata(obs, X, self.var_names.tolist(), obsm, layers)
        obs = self.obs.loc[item].copy()
        positions = [self.obs.index.get_loc(idx) for idx in obs.index]
        X = self.X[positions].copy()
        obsm = {key: value[positions].copy() for key, value in self.obsm.items()}
        layers = {key: value[positions].copy() for key, value in self.layers.items()}
        return FakeAdata(obs, X, self.var_names.tolist(), obsm, layers)

    def write(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("adata", encoding="utf-8")


def build_embeddings_dict(tmp_path: Path) -> dict:
    run_dir = create_run_directory(
        root_dir=tmp_path,
        category="embeddings",
        dataset_name="scrna",
        model_name="demo_model",
        timestamp="2026-04-01T10-00-00",
    )

    cell_df = pd.DataFrame(
        [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3]],
        index=pd.Index(["c1", "c2", "c3", "c4"], name="cell_id"),
    )
    centroid_df = pd.DataFrame(
        [[1.0, 0.0], [0.0, 1.0]],
        index=pd.Index(["Forebrain", "Dorsal forebrain"], name="cell_id"),
    )

    cell_meta = save_embedding_frame(
        run_dir,
        "df_cells",
        cell_df,
        annotation_map={
            "celltype": {
                "c1": "Forebrain",
                "c2": "Forebrain",
                "c3": "Dorsal forebrain",
                "c4": "Dorsal forebrain",
            }
        },
    )
    cell_meta["run_dir"] = str(run_dir)

    centroid_meta = save_embedding_frame(
        run_dir,
        "df_celltypes",
        centroid_df,
        annotation_map={
            "Forebrain": "Forebrain",
            "Dorsal forebrain": "Dorsal forebrain",
        },
    )
    centroid_meta["run_dir"] = str(run_dir)

    return {"demo_model": {"scrna": {"df_cells": cell_meta, "df_celltypes": centroid_meta}}}


def build_adata() -> FakeAdata:
    obs = pd.DataFrame(
        {
            "celltype": ["Forebrain", "Forebrain", "Dorsal forebrain", "Dorsal forebrain"],
            "time": [0.0, 1.0, 2.0, 3.0],
            "sample": ["s1", "s1", "s2", "s2"],
        },
        index=["c1", "c2", "c3", "c4"],
    )
    X = np.array(
        [
            [1.0, 0.2],
            [0.9, 0.3],
            [0.8, 0.4],
            [0.7, 0.5],
        ],
        dtype=float,
    )
    return FakeAdata(obs=obs, X=X, var_names=["Pax6", "Fezf1"], layers={"counts": X.copy()})


def test_canonicalize_centroid_labels_removes_duplicate_cell_type_columns(monkeypatch):
    module = load_pseudotime_module(monkeypatch)
    df = pd.DataFrame(
        {
            "celltype": ["Forebrain", "Dorsal forebrain"],
            "cell_type": ["old Forebrain", "old Dorsal forebrain"],
            "UMAP1": [0.0, 1.0],
            "UMAP2": [0.5, 1.5],
        }
    )

    result = module._canonicalize_centroid_labels(df, "celltype")

    assert result.columns.tolist().count("cell_type") == 1
    assert "celltype" not in result.columns
    assert result["cell_type"].tolist() == ["Forebrain", "Dorsal forebrain"]
    assert {"UMAP1", "UMAP2"} <= set(result.columns)


def test_resolve_precomputed_subset_umap_returns_selected_lineage(monkeypatch):
    module = load_pseudotime_module(monkeypatch)
    expected = {"cells": pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]})}
    precomputed = {"demo_model": {"scrna": {"lineage_4": expected}}}

    result = module._resolve_precomputed_subset_umap(
        precomputed,
        saved_model_name="demo_model",
        dataset_name="scrna",
        lineage_name="lineage_4",
    )

    assert result is expected


def test_pseudotime_writes_expected_artifacts(tmp_path: Path, monkeypatch):
    module = load_pseudotime_module(monkeypatch)
    monkeypatch.setattr(module, "UMAPCellPlotter", DummyPlotter, raising=False)
    monkeypatch.setattr(
        module,
        "_cal_umap",
        lambda df_full, evaluation_config: df_full.assign(
            UMAP1=np.arange(len(df_full), dtype=float),
            UMAP2=np.arange(len(df_full), dtype=float) + 0.25,
        ),
        raising=False,
    )

    results = module.pseudotime(
        embeddings_dict=build_embeddings_dict(tmp_path),
        eval_data={"adata": build_adata()},
        subfolder_fig_dir=tmp_path / "evaluation_plots",
        annotation_column="celltype",
        evaluation_config=types.SimpleNamespace(n_neighbors=5, min_dist=0.5, n_components=2, random_state=73),
        config=module.PseudotimeConfig(lineage="lineage_4", cell_origin="Forebrain"),
    )

    run_dir = (
        tmp_path
        / "evaluation_plots"
        / "demo_model"
        / "scrna"
        / "pseudotime"
        / "2026-04-01T10-00-00"
    )
    lineage_dir = run_dir / "lineage_4"

    assert run_dir.exists()
    assert lineage_dir.exists()
    assert (run_dir / "metadata.json").exists()
    assert (lineage_dir / "pseudotime_comparison_time_lineage_4.pdf").exists()
    assert (lineage_dir / "pseudotime_comparison_celltypes_lineage_4.pdf").exists()
    assert (lineage_dir / "adata_forebrain_llm_20260120.h5ad").exists()
    assert (lineage_dir / "pseudotime_values.csv").exists()
    assert isinstance(results, pd.DataFrame)
    assert {
        "cell_id",
        "celltype",
        "time",
        "dpt_pseudotime_llm",
        "dpt_pseudotime_expr",
        "dpt_pseudotime_scvi",
    } <= set(results.columns)
    assert len(results) == 4

    saved_values = pd.read_csv(lineage_dir / "pseudotime_values.csv", index_col=0)
    assert {
        "celltype",
        "time",
        "dpt_pseudotime_llm",
        "dpt_pseudotime_expr",
        "dpt_pseudotime_scvi",
    } <= set(saved_values.columns)
    assert saved_values["dpt_pseudotime_llm"].notna().all()
    assert saved_values["dpt_pseudotime_expr"].notna().all()
    assert saved_values["dpt_pseudotime_scvi"].notna().all()


def test_pseudotime_adata_umap_uses_fixed_scanpy_settings(tmp_path: Path, monkeypatch):
    module = load_pseudotime_module(monkeypatch)
    monkeypatch.setattr(module, "UMAPCellPlotter", DummyPlotter, raising=False)
    scanpy_calls = {"pca": [], "neighbors": [], "umap": [], "diffmap": [], "dpt": []}

    def record_pca(adata, *args, **kwargs):
        scanpy_calls["pca"].append(kwargs)
        n_rows = len(adata.obs.index)
        adata.obsm["X_pca"] = np.column_stack(
            [np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 1.0]
        )

    def record_neighbors(adata, *args, **kwargs):
        scanpy_calls["neighbors"].append(kwargs)

    def record_diffmap(adata, *args, **kwargs):
        scanpy_calls["diffmap"].append(kwargs)

    def record_dpt(adata, *args, **kwargs):
        scanpy_calls["dpt"].append(kwargs)
        n_rows = len(adata.obs.index)
        adata.obs["dpt_pseudotime"] = np.linspace(0.0, 1.0, n_rows)

    def record_umap(adata, *args, **kwargs):
        scanpy_calls["umap"].append(kwargs)
        n_rows = len(adata.obs.index)
        adata.obsm["X_umap"] = np.column_stack(
            [np.arange(n_rows, dtype=float), np.arange(n_rows, dtype=float) + 0.5]
        )

    monkeypatch.setattr(module.sc.tl, "pca", record_pca, raising=False)
    monkeypatch.setattr(module.sc.pp, "neighbors", record_neighbors, raising=False)
    monkeypatch.setattr(module.sc.tl, "umap", record_umap, raising=False)
    monkeypatch.setattr(module.sc.tl, "diffmap", record_diffmap, raising=False)
    monkeypatch.setattr(module.sc.tl, "dpt", record_dpt, raising=False)
    monkeypatch.setattr(
        module,
        "_cal_umap",
        lambda df_full, evaluation_config: df_full.assign(
            UMAP1=np.arange(len(df_full), dtype=float),
            UMAP2=np.arange(len(df_full), dtype=float) + 0.25,
        ),
        raising=False,
    )

    module.pseudotime(
        embeddings_dict=build_embeddings_dict(tmp_path),
        eval_data={"adata": build_adata()},
        subfolder_fig_dir=tmp_path / "evaluation_plots",
        annotation_column="celltype",
        evaluation_config=types.SimpleNamespace(n_neighbors=3, min_dist=0.7, n_components=2, random_state=99),
        config=module.PseudotimeConfig(lineage="lineage_4", cell_origin="Forebrain"),
    )

    assert scanpy_calls["neighbors"][0] == {"use_rep": "X_llm", "n_neighbors": 50}
    assert scanpy_calls["pca"][0] == {"n_comps": 50, "svd_solver": "arpack"}
    assert scanpy_calls["neighbors"][1] == {"n_neighbors": 50, "use_rep": "X_pca"}
    assert scanpy_calls["neighbors"][2] == {"n_neighbors": 15, "use_rep": "X_scVI"}
    assert scanpy_calls["umap"] == [{"min_dist": 0.5, "spread": 0.5, "random_state": module.SEED}]
    assert scanpy_calls["diffmap"] == [{}, {}]
    assert len(scanpy_calls["dpt"]) == 3
    scvi_model = module.scvi.model.SCVI
    assert scvi_model.setup_calls[0]["kwargs"] == {"layer": "counts", "batch_key": "sample"}
    assert scvi_model.train_calls[0] == {
        "max_epochs": 200,
        "early_stopping": True,
        "early_stopping_patience": 10,
        "batch_size": 256,
    }
