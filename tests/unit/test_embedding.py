import numpy as np

from sot_matcher.infrastructure.embedding import SentenceTransformerEmbedder


class CountingDelegate:
    def __init__(self) -> None:
        self.calls = 0
        self.seen: list[list[str]] = []

    def encode(self, sentences: list[str], **_kwargs: object) -> np.ndarray:
        self.calls += 1
        self.seen.append(list(sentences))
        return np.asarray([[1.0, 0.0] if "talara" in sentence else [0.0, 1.0] for sentence in sentences])


def test_encode_caches_vectors_and_skips_known_texts():
    delegate = CountingDelegate()
    embedder = SentenceTransformerEmbedder(delegate=delegate)

    first = embedder.encode(["talara norte", "otro texto"], normalize_embeddings=True, batch_size=64)
    second = embedder.encode(["talara norte", "otro texto", "tercero"], normalize_embeddings=True, batch_size=64)

    assert delegate.calls == 2
    assert delegate.seen[1] == ["tercero"]
    np.testing.assert_allclose(second[0], first[0])
    np.testing.assert_allclose(second[1], first[1])


def test_encode_preserves_order_and_duplicates():
    delegate = CountingDelegate()
    embedder = SentenceTransformerEmbedder(delegate=delegate)

    result = embedder.encode(["b", "a", "b"], normalize_embeddings=True)
    assert result.shape == (3, 2)
    np.testing.assert_allclose(result[0], result[2])

    again = embedder.encode(["a", "b"], normalize_embeddings=True)
    assert delegate.calls == 1  # todo viene de caché
    np.testing.assert_allclose(again[1], result[0])


def test_encode_with_different_normalization_is_not_reused():
    delegate = CountingDelegate()
    embedder = SentenceTransformerEmbedder(delegate=delegate)

    embedder.encode(["x"], normalize_embeddings=True)
    embedder.encode(["x"])
    assert delegate.calls == 2