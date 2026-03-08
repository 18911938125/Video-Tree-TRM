"""
嵌入服务模块单元测试
====================
覆盖: 本地模式（embed/embed_tensor/归一化/dim/冻结）、远程模式（真实 API 调用）。

本地测试使用轻量模型 all-MiniLM-L6-v2 (dim=384) 加速。
远程测试使用 .env 中配置的真实 API（需有效密钥）。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from video_tree_trm.config import Config, EmbedConfig
from video_tree_trm.embeddings import EmbeddingModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_LOCAL_CONFIG = EmbedConfig(
    backend="local",
    model_name="all-MiniLM-L6-v2",
    embed_dim=384,
    device="cpu",
    api_key="",
    api_url="",
)


@pytest.fixture(scope="module")
def local_model() -> EmbeddingModel:
    """本地嵌入模型（模块级缓存，避免重复加载）。"""
    return EmbeddingModel(_LOCAL_CONFIG)


@pytest.fixture(scope="module")
def remote_model(real_config: Config) -> EmbeddingModel:
    """远程嵌入模型（模块级缓存），使用真实 API。"""
    return EmbeddingModel(real_config.embed)


# ---------------------------------------------------------------------------
# 测试: 本地模式
# ---------------------------------------------------------------------------


class TestLocalEmbed:
    """本地 sentence-transformers 后端测试。"""

    def test_embed_single_text(self, local_model: EmbeddingModel) -> None:
        """单条文本，验证形状 [1, D]。"""
        result = local_model.embed("hello world")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 384)

    def test_embed_batch(self, local_model: EmbeddingModel) -> None:
        """批量文本，验证形状 [N, D]。"""
        texts = ["hello", "world", "test"]
        result = local_model.embed(texts)
        assert result.shape == (3, 384)

    def test_embed_tensor_type(self, local_model: EmbeddingModel) -> None:
        """验证 embed_tensor() 返回 Tensor。"""
        result = local_model.embed_tensor("hello")
        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 384)
        assert result.dtype == torch.float32

    def test_embeddings_normalized(self, local_model: EmbeddingModel) -> None:
        """验证 L2 范数 ≈ 1.0。"""
        result = local_model.embed(["hello", "world"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_dim_property(self, local_model: EmbeddingModel) -> None:
        """验证 dim 属性一致。"""
        assert local_model.dim == 384

    def test_local_model_frozen(self, local_model: EmbeddingModel) -> None:
        """本地模式参数 requires_grad=False。"""
        for param in local_model._model.parameters():
            assert not param.requires_grad


# ---------------------------------------------------------------------------
# 测试: 远程模式（真实 API）
# ---------------------------------------------------------------------------


class TestRemoteEmbed:
    """远程 OpenAI 兼容 API 后端测试（真实 API 调用）。"""

    def test_remote_embed_single(self, remote_model: EmbeddingModel) -> None:
        """单条中文文本，验证形状 [1, D]。"""
        result = remote_model.embed("你好世界")
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, remote_model.dim)

    def test_remote_embed_batch(self, remote_model: EmbeddingModel) -> None:
        """5 条文本，验证形状 [5, D]。"""
        texts = ["机器学习", "深度学习", "自然语言处理", "计算机视觉", "强化学习"]
        result = remote_model.embed(texts)
        assert result.shape == (5, remote_model.dim)

    def test_remote_embed_normalized(self, remote_model: EmbeddingModel) -> None:
        """验证 L2 范数 ≈ 1.0。"""
        result = remote_model.embed(["归一化测试文本", "另一段测试文本"])
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_remote_embed_tensor(self, remote_model: EmbeddingModel) -> None:
        """验证返回 torch.Tensor + float32。"""
        result = remote_model.embed_tensor("张量测试")
        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.float32
        assert result.shape == (1, remote_model.dim)

    def test_remote_dim(self, remote_model: EmbeddingModel, real_config: Config) -> None:
        """验证 dim 属性与配置一致。"""
        assert remote_model.dim == real_config.embed.embed_dim

    def test_remote_semantic_similarity(self, remote_model: EmbeddingModel) -> None:
        """语义相近文本相似度 > 语义无关文本。"""
        # 语义相近对
        vec_cat = remote_model.embed("猫咪在沙发上睡觉")
        vec_kitten = remote_model.embed("小猫趴在沙发上打盹")
        # 语义无关
        vec_math = remote_model.embed("二次方程的求解公式")

        sim_close = float(np.dot(vec_cat[0], vec_kitten[0]))
        sim_far = float(np.dot(vec_cat[0], vec_math[0]))
        assert sim_close > sim_far, (
            f"语义相近对相似度 ({sim_close:.4f}) 应大于语义无关对 ({sim_far:.4f})"
        )


# ---------------------------------------------------------------------------
# 测试: 配置校验
# ---------------------------------------------------------------------------


class TestConfigValidation:
    """配置校验测试。"""

    def test_invalid_backend(self) -> None:
        """无效 backend 应抛出 ValueError。"""
        config = EmbedConfig(
            backend="invalid",
            model_name="test",
            embed_dim=384,
            device="cpu",
            api_key="",
            api_url="",
        )
        with pytest.raises(ValueError, match="backend"):
            EmbeddingModel(config)

    def test_remote_missing_api_key(self) -> None:
        """远程模式缺少 api_key 应抛出 ValueError。"""
        config = EmbedConfig(
            backend="remote",
            model_name="test",
            embed_dim=384,
            device="cpu",
            api_key="",
            api_url="http://localhost:8080/v1",
        )
        with pytest.raises(ValueError, match="api_key"):
            EmbeddingModel(config)

    def test_remote_missing_api_url(self) -> None:
        """远程模式缺少 api_url 应抛出 ValueError。"""
        config = EmbedConfig(
            backend="remote",
            model_name="test",
            embed_dim=384,
            device="cpu",
            api_key="test-key",
            api_url="",
        )
        with pytest.raises(ValueError, match="api_url"):
            EmbeddingModel(config)

    def test_local_dim_mismatch(self) -> None:
        """本地模式维度不一致应抛出 ValueError。"""
        config = EmbedConfig(
            backend="local",
            model_name="all-MiniLM-L6-v2",
            embed_dim=999,  # 实际是 384
            device="cpu",
            api_key="",
            api_url="",
        )
        with pytest.raises(ValueError, match="维度"):
            EmbeddingModel(config)
