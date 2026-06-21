"""Builds a CascadePipeline from an experiment config by instantiating each
named stage from its model config. Stages register themselves into STAGE_REGISTRY;
adding a new stage type means adding one builder function here.
"""

from cascade.pipeline.cascade import CascadePipeline
from cascade.stages.bi_encoder_stage import BiEncoderStage
from cascade.stages.bm25_stage import BM25Stage
from cascade.stages.cross_encoder_stage import CrossEncoderStage
from cascade.utils.io import load_yaml

STAGE_REGISTRY = {}


def register_stage(name):
    def deco(builder):
        STAGE_REGISTRY[name] = builder
        return builder

    return deco


@register_stage("bm25")
def _build_bm25(model_cfg: dict, dataset_cfg: dict, label_texts: list[str]):
    trunc = dataset_cfg.get("truncation", {})
    params = model_cfg.get("params", {})
    return BM25Stage(
        label_texts,
        k1=params.get("k1", 1.5),
        b=params.get("b", 0.75),
        query_truncate_tokens=trunc.get("bm25_query_tokens", 512),
    )


@register_stage("bi_encoder")
def _build_bi_encoder(model_cfg: dict, dataset_cfg: dict, label_texts: list[str]):
    trunc = dataset_cfg.get("truncation", {})
    return BiEncoderStage(
        label_texts,
        model_name=model_cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        device=model_cfg.get("device", "cuda"),
        fp16=model_cfg.get("fp16", True),
        batch_size=model_cfg.get("batch_size", 256),
        doc_truncate_tokens=trunc.get("encoder_doc_tokens", 384),
    )


@register_stage("cross_encoder")
def _build_cross_encoder(model_cfg: dict, dataset_cfg: dict, label_texts: list[str]):
    return CrossEncoderStage(
        label_texts,
        model_name=model_cfg.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        device=model_cfg.get("device", "cuda"),
        fp16=model_cfg.get("fp16", True),
        batch_size=model_cfg.get("batch_size", 64),
        max_length=model_cfg.get("max_length", 512),
    )


def build_pipeline(
    exp_cfg: dict, dataset_cfg: dict, label_texts_by_mode: dict[bool, list[str] | None]
) -> CascadePipeline:
    """label_texts_by_mode: {False: raw label texts, True: RAG-augmented label
    texts or None if unavailable for this dataset}. Each pipeline step can set
    its own `use_rag_labels` to override the experiment-level default — e.g.
    BM25 stays lexical (raw label text) while bi/cross-encoder stages use the
    RAG-augmented text, within the same pipeline.
    """
    default_use_rag = exp_cfg.get("use_rag_labels", False)
    stages = []
    for step in exp_cfg["pipeline"]:
        stage_name = step["stage"]
        if stage_name not in STAGE_REGISTRY:
            raise NotImplementedError(
                f"Stage '{stage_name}' not yet implemented. Available: {list(STAGE_REGISTRY)}"
            )
        use_rag = step.get("use_rag_labels", default_use_rag)
        label_texts = label_texts_by_mode.get(use_rag)
        if label_texts is None:
            raise FileNotFoundError(
                f"Stage '{stage_name}' requests use_rag_labels=true but RAG-labels "
                f"aren't available for dataset '{dataset_cfg['name']}'. Run "
                f"scripts/05_generate_rag_labels.py --dataset {dataset_cfg['name']} first."
            )
        model_config_path = step.get("model_config", f"configs/models/{stage_name}.yaml")
        model_cfg = load_yaml(model_config_path)
        stage = STAGE_REGISTRY[stage_name](model_cfg, dataset_cfg, label_texts)
        stages.append((stage, step["top_k"]))
    return CascadePipeline(stages, batch_size=exp_cfg.get("batch_size", 64))
