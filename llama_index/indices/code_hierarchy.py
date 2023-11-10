from collections import defaultdict
from typing import Any, Dict, List, Sequence, Set, Union

from llama_index import VectorStoreIndex
from llama_index.data_structs.data_structs import KeywordTable
from llama_index.embeddings.base import BaseEmbedding
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.keyword_table.base import (
    BaseKeywordTableIndex,
    KeywordTableRetrieverMode,
)
from llama_index.schema import BaseNode, MetadataMode
from llama_index.utils import get_tqdm_iterable


class CodeHierarchyKeywordTableIndex(BaseKeywordTableIndex):
    """A keyword table made specifically to work with the code hierarchy node parser.

    Similar to SimpleKeywordTableIndex, but doesn't use GPT to extract keywords.
    """

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        raise NotImplementedError(
            "You should not be calling this method "
            "from within CodeHierarchyKeywordTableIndex."
        )

    def _extract_keywords_from_node(self, node: BaseNode) -> Set[str]:
        keywords = []
        keywords.append(node.node_id)
        file_path = node.metadata["filepath"]
        module_path = file_path.replace("/", ".").lstrip(".").rstrip(".py")
        keywords.append(module_path)
        # Add the last scope name and signature to the keywords
        if node.metadata["inclusive_scopes"]:
            keywords.append(node.metadata["inclusive_scopes"][-1]["name"])
            keywords.append(node.metadata["inclusive_scopes"][-1]["signature"])

        return {k.lower() for k in keywords}

    def _add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        nodes_with_progress = get_tqdm_iterable(
            nodes, show_progress, "Extracting keywords from nodes"
        )
        for n in nodes_with_progress:
            index_struct.add_node(list(self._extract_keywords_from_node(n)), n)

    async def _async_add_nodes_to_index(
        self,
        index_struct: KeywordTable,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        """Add document to index."""
        return self._add_nodes_to_index(index_struct, nodes, show_progress)

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Insert nodes."""
        for n in nodes:
            self._index_struct.add_node(list(self._extract_keywords_from_node(n)), n)

    def as_retriever(
        self,
        retriever_mode: Union[str, KeywordTableRetrieverMode, None] = None,
        **kwargs: Any,
    ) -> BaseRetriever:
        if retriever_mode is None:
            from llama_index.indices.keyword_table.base import KeywordTableRetrieverMode

            retriever_mode = KeywordTableRetrieverMode.SIMPLE
        return super().as_retriever(retriever_mode=retriever_mode, **kwargs)


def _get_id_to_embed_map(nodes):
    id_to_embed_map: Dict[str, List[List[float]]] = defaultdict(list)

    texts_to_embed = []
    ids_to_embed = []

    for node in nodes:
        # Embed the text itself
        ids_to_embed.append(node.node_id)
        texts_to_embed.append(node.get_content(metadata_mode=MetadataMode.EMBED))

        # Embed the uuid
        ids_to_embed.append(node.node_id)
        texts_to_embed.append(node.node_id)

        # Embed the file path
        file_path = node.metadata["filepath"]
        module_path = file_path.replace("/", ".").lstrip(".").rstrip(".py")
        ids_to_embed.append(node.node_id)
        texts_to_embed.append(module_path)

        # Embed the last scope name and signature
        if node.metadata["inclusive_scopes"]:
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.metadata["inclusive_scopes"][-1]["name"])
            ids_to_embed.append(node.node_id)
            texts_to_embed.append(node.metadata["inclusive_scopes"][-1]["signature"])
    return id_to_embed_map, texts_to_embed, ids_to_embed


async def _async_embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> Dict[str, List[List[float]]]:
    """Async get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    """
    id_to_embed_map, texts_to_embed, ids_to_embed = _get_id_to_embed_map(nodes)

    new_embeddings = await embed_model.aget_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id].append(text_embedding)

    return id_to_embed_map


def _embed_nodes(
    nodes: Sequence[BaseNode], embed_model: BaseEmbedding, show_progress: bool = False
) -> Dict[str, List[List[float]]]:
    """Get embeddings of the given nodes, run embedding model if necessary.

    Args:
        nodes (Sequence[BaseNode]): The nodes to embed.
        embed_model (BaseEmbedding): The embedding model to use.
        show_progress (bool): Whether to show progress bar.

    """
    id_to_embed_map, texts_to_embed, ids_to_embed = _get_id_to_embed_map(nodes)

    new_embeddings = embed_model.get_text_embedding_batch(
        texts_to_embed, show_progress=show_progress
    )

    for new_id, text_embedding in zip(ids_to_embed, new_embeddings):
        id_to_embed_map[new_id].append(text_embedding)

    return id_to_embed_map


class CodeHierarchyVectorStoreIndex(VectorStoreIndex):
    """Vector Store Index.

    Args:
        use_async (bool): Whether to use asynchronous calls. Defaults to False.
        show_progress (bool): Whether to show tqdm progress bars. Defaults to False.
        store_nodes_override (bool): set to True to always store Node objects in index
            store and document store even if vector store keeps text. Defaults to False
    """

    def _get_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = _embed_nodes(
            nodes=nodes,
            embed_model=self._service_context.embed_model,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embeddings = id_to_embed_map[node.node_id]
            for embedding in embeddings:
                result = node.copy()
                result.embedding = embedding
                results.append(result)
        return results

    async def _aget_node_with_embedding(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[BaseNode]:
        """Asynchronously get tuples of id, node, and embedding.

        Allows us to store these nodes in a vector store.
        Embeddings are called in batches.

        """
        id_to_embed_map = await _async_embed_nodes(
            nodes=nodes,
            embed_model=self._service_context.embed_model,
            show_progress=show_progress,
        )

        results = []
        for node in nodes:
            embeddings = id_to_embed_map[node.node_id]
            for embedding in embeddings:
                result = node.copy()
                result.embedding = embedding
                results.append(result)
        return results
