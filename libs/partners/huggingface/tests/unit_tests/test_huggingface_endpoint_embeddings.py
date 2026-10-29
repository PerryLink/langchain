"""Tests for `HuggingFaceEndpointEmbeddings` (no Hugging Face API calls)."""

from unittest.mock import MagicMock, patch

from langchain_core.load import dumpd

from langchain_huggingface.embeddings.huggingface_endpoint import (
    HuggingFaceEndpointEmbeddings,
)

API_TOKEN = "hf_SUPERSECRET123"  # noqa: S105


@patch("huggingface_hub.AsyncInferenceClient")
@patch("huggingface_hub.InferenceClient")
def test_api_token_not_in_repr_or_serialized(
    mock_inference_client: MagicMock,
    mock_async_client: MagicMock,
) -> None:
    """The API token must not leak through `repr`, `str` or `dumpd`."""
    mock_inference_client.return_value = MagicMock()
    mock_async_client.return_value = MagicMock()

    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-mpnet-base-v2",
        huggingfacehub_api_token=API_TOKEN,
    )

    assert API_TOKEN not in repr(embeddings)
    assert API_TOKEN not in str(embeddings)
    # `dumpd` falls back to `repr` for objects that are not LangChain-serializable.
    assert API_TOKEN not in str(dumpd(embeddings))

    # The field keeps its type and is still handed to both clients.
    assert embeddings.huggingfacehub_api_token == API_TOKEN
    assert mock_inference_client.call_args[1]["token"] == API_TOKEN
    assert mock_async_client.call_args[1]["token"] == API_TOKEN
