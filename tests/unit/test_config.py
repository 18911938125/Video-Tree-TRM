"""
配置管理模块单元测试
====================
覆盖: YAML 加载、.env 覆盖、CLI 覆盖、缺字段报错、优先级验证、embed_dim 一致性。
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from video_tree_trm.config import Config, TreeConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FULL_YAML = {
    "tree": {
        "max_paragraphs_per_l2": 5,
        "l1_segment_duration": 600.0,
        "l2_clip_duration": 20.0,
        "l3_fps": 1.0,
        "l2_representative_frames": 3,
        "cache_dir": "cache/trees",
    },
    "embed": {
        "backend": "local",
        "model_name": "test-model",
        "embed_dim": 2560,
        "device": "cpu",
        "api_key": "",
        "api_url": "",
    },
    "llm": {
        "backend": "qwen",
        "api_key": "yaml-llm-key",
        "model": "qwen-plus",
        "api_url": "https://example.com/llm",
        "max_tokens": 256,
        "temperature": 0.1,
    },
    "vlm": {
        "backend": "qwen",
        "api_key": "yaml-vlm-key",
        "model": "qwen-vl-plus",
        "api_url": "https://example.com/vlm",
        "max_tokens": 256,
        "temperature": 0.1,
    },
    "retriever": {
        "embed_dim": 2560,
        "num_heads": 4,
        "L_layers": 2,
        "L_cycles": 4,
        "max_rounds": 5,
        "ffn_expansion": 2.0,
        "checkpoint": None,
    },
    "train": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 1,
        "max_epochs_phase1": 30,
        "max_epochs_phase2": 20,
        "nav_loss_weight": 1.0,
        "act_loss_weight": 0.1,
        "act_lambda_step": 0.1,
        "act_gamma": 0.9,
        "eval_interval": 5,
        "save_dir": "checkpoints",
        "dataset": "longbench",
        "dataset_path": "data/longbench",
    },
}


@pytest.fixture()
def yaml_path(tmp_path: Path) -> Path:
    """创建完整配置的临时 YAML 文件。"""
    p = tmp_path / "config" / "default.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(_FULL_YAML, f, allow_unicode=True)
    return p


@pytest.fixture()
def env_path(tmp_path: Path) -> Path:
    """创建临时 .env 文件。"""
    p = tmp_path / ".env"
    p.write_text(
        "LLM_API_KEY=env-llm-key\n"
        "LLM_MODEL=env-llm-model\n"
        "LLM_API_URL=https://env.example.com/llm\n"
        "VLM_API_KEY=env-vlm-key\n"
        "VLM_MODEL=env-vlm-model\n"
        "VLM_API_URL=https://env.example.com/vlm\n"
        "EMBED_BACKEND=remote\n"
        "EMBED_MODEL=env-embed-model\n"
        "EMBED_API_KEY=env-embed-key\n"
        "EMBED_API_URL=https://env.example.com/embed\n"
    )
    return p


# ---------------------------------------------------------------------------
# 测试: YAML 加载
# ---------------------------------------------------------------------------


class TestYAMLLoad:
    """YAML 基础加载测试。"""

    def test_load_full_yaml(self, yaml_path: Path) -> None:
        """完整 YAML 应成功加载所有字段。"""
        cfg = Config.load(
            str(yaml_path), env_path=str(yaml_path.parent / ".env.nonexist")
        )
        assert isinstance(cfg.tree, TreeConfig)
        assert cfg.tree.max_paragraphs_per_l2 == 5
        assert cfg.tree.l1_segment_duration == 600.0
        assert cfg.embed.embed_dim == 2560
        assert cfg.retriever.checkpoint is None
        assert cfg.train.dataset == "longbench"

    def test_file_not_found(self, tmp_path: Path) -> None:
        """不存在的 YAML 应抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="配置文件不存在"):
            Config.load(str(tmp_path / "nonexist.yaml"))


# ---------------------------------------------------------------------------
# 测试: 缺字段报错
# ---------------------------------------------------------------------------


class TestMissingField:
    """缺少必需字段时应抛出 TypeError。"""

    def test_missing_tree_field(self, tmp_path: Path) -> None:
        """tree 节缺少字段应报 TypeError。"""
        bad_yaml = _FULL_YAML.copy()
        bad_yaml = {
            k: (v.copy() if isinstance(v, dict) else v) for k, v in _FULL_YAML.items()
        }
        del bad_yaml["tree"]["cache_dir"]

        p = tmp_path / "bad.yaml"
        with open(p, "w") as f:
            yaml.dump(bad_yaml, f)

        with pytest.raises(TypeError):
            Config.load(str(p), env_path=str(tmp_path / ".env.nonexist"))

    def test_missing_section(self, tmp_path: Path) -> None:
        """缺少整个配置节应报 TypeError。"""
        bad_yaml = {k: v for k, v in _FULL_YAML.items() if k != "train"}

        p = tmp_path / "bad2.yaml"
        with open(p, "w") as f:
            yaml.dump(bad_yaml, f)

        with pytest.raises(TypeError):
            Config.load(str(p), env_path=str(tmp_path / ".env.nonexist"))


# ---------------------------------------------------------------------------
# 测试: .env 覆盖
# ---------------------------------------------------------------------------


class TestEnvOverride:
    """.env 文件应覆盖 YAML 中的 api_key。"""

    def test_env_overrides_api_keys(self, yaml_path: Path, env_path: Path) -> None:
        """api_key/model/api_url 应优先使用 .env 中的值。"""
        cfg = Config.load(str(yaml_path), env_path=str(env_path))
        assert cfg.llm.api_key == "env-llm-key"
        assert cfg.llm.model == "env-llm-model"
        assert cfg.llm.api_url == "https://env.example.com/llm"
        assert cfg.vlm.api_key == "env-vlm-key"
        assert cfg.vlm.model == "env-vlm-model"
        assert cfg.vlm.api_url == "https://env.example.com/vlm"

    def test_env_overrides_embed(self, yaml_path: Path, env_path: Path) -> None:
        """embed 相关字段应优先使用 .env 中的值。"""
        cfg = Config.load(str(yaml_path), env_path=str(env_path))
        assert cfg.embed.backend == "remote"
        assert cfg.embed.model_name == "env-embed-model"
        assert cfg.embed.api_key == "env-embed-key"
        assert cfg.embed.api_url == "https://env.example.com/embed"

    def test_yaml_fallback_when_no_env(self, yaml_path: Path) -> None:
        """无 .env 时应使用 YAML 中的值。"""
        cfg = Config.load(
            str(yaml_path), env_path=str(yaml_path.parent / ".env.nonexist")
        )
        assert cfg.llm.api_key == "yaml-llm-key"
        assert cfg.vlm.api_key == "yaml-vlm-key"


# ---------------------------------------------------------------------------
# 测试: CLI 覆盖
# ---------------------------------------------------------------------------


class TestCLIOverride:
    """CLI args 应覆盖 YAML 和 .env 的值。"""

    def test_cli_overrides_yaml(self, yaml_path: Path) -> None:
        """CLI 点路径覆盖应生效。"""
        cfg = Config.load(
            str(yaml_path),
            cli_args={"retriever.num_heads": 8, "train.lr": 0.001},
            env_path=str(yaml_path.parent / ".env.nonexist"),
        )
        assert cfg.retriever.num_heads == 8
        assert cfg.train.lr == 0.001

    def test_cli_overrides_env(self, yaml_path: Path, env_path: Path) -> None:
        """CLI 应覆盖 .env 中的 api_key。"""
        cfg = Config.load(
            str(yaml_path),
            cli_args={"llm.api_key": "cli-key"},
            env_path=str(env_path),
        )
        assert cfg.llm.api_key == "cli-key"


# ---------------------------------------------------------------------------
# 测试: 优先级
# ---------------------------------------------------------------------------


class TestPriority:
    """三层优先级: CLI > .env > YAML。"""

    def test_full_priority_chain(self, yaml_path: Path, env_path: Path) -> None:
        """CLI > .env > YAML 的完整优先级链。"""
        cfg = Config.load(
            str(yaml_path),
            cli_args={"llm.api_key": "cli-key"},
            env_path=str(env_path),
        )
        # CLI 覆盖 .env
        assert cfg.llm.api_key == "cli-key"
        # .env 覆盖 YAML（vlm 未被 CLI 覆盖）
        assert cfg.vlm.api_key == "env-vlm-key"


# ---------------------------------------------------------------------------
# 测试: embed_dim 一致性校验
# ---------------------------------------------------------------------------


class TestEmbedDimConsistency:
    """embed.embed_dim 与 retriever.embed_dim 必须一致。"""

    def test_inconsistent_embed_dim(self, tmp_path: Path) -> None:
        """embed_dim 不一致应抛出 ValueError。"""
        bad_yaml = {
            k: (v.copy() if isinstance(v, dict) else v) for k, v in _FULL_YAML.items()
        }
        bad_yaml["retriever"] = bad_yaml["retriever"].copy()
        bad_yaml["retriever"]["embed_dim"] = 512  # 与 embed.embed_dim=768 不一致

        p = tmp_path / "bad_dim.yaml"
        with open(p, "w") as f:
            yaml.dump(bad_yaml, f)

        with pytest.raises(ValueError, match="不一致"):
            Config.load(str(p), env_path=str(tmp_path / ".env.nonexist"))
