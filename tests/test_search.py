import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from rag_search_engine import config
from rag_search_engine.core import (
    cross_encoder,
    inverted_index,
    llm,
    multimodal_search,
    semantic_search,
)
from cli import augmented_generation_cli, semantic_search_cli
from rag_search_engine.core.hybrid_search import HybridSearch, hybrid_score, rrf_score
from rag_search_engine.utils.chunking import (
    chunk_by_text_sentences,
    chunk_text_by_words,
)


class ChunkingTests(unittest.TestCase):
    def test_word_overlap(self):
        self.assertEqual(
            chunk_text_by_words("one two three four", 3, 1),
            ["one two three", "three four"],
        )

    def test_sentence_overlap_does_not_repeat_final_chunk(self):
        self.assertEqual(
            chunk_by_text_sentences("One. Two. Three. Four.", 3, 1),
            ["One. Two. Three.", "Three. Four."],
        )

    def test_empty_text(self):
        self.assertEqual(chunk_text_by_words("", 3), [])
        self.assertEqual(chunk_by_text_sentences("", 3, 1), [])

    def test_invalid_overlap(self):
        for overlap in (-1, 3):
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                chunk_text_by_words("text", 3, overlap)
            with self.subTest(overlap=overlap), self.assertRaises(ValueError):
                chunk_by_text_sentences("text", 3, overlap)


class IndexTests(unittest.TestCase):
    def setUp(self):
        tokenizer = patch.object(inverted_index, "tokenize_text", side_effect=str.split)
        tokenizer.start()
        self.addCleanup(tokenizer.stop)
        self.index = inverted_index.InvertedIndex()

    def test_build_uses_supplied_corpus_and_preserves_context(self):
        description = "alpha " * 30
        documents = [{"id": 9, "title": "alpha", "description": description}]
        with patch.object(inverted_index, "load_movies") as loader:
            self.index.build(documents)
        loader.assert_not_called()
        result = self.index.bm25_search("alpha")[0]
        self.assertEqual(result["id"], 9)
        self.assertEqual(result["description"], description)

    def test_rebuild_removes_old_terms_and_resets_average_length(self):
        self.index.build([{"id": 1, "title": "alpha", "description": "alpha alpha"}])
        self.assertEqual(self.index._get_avg_doc_length(), 3)
        self.index.build([{"id": 2, "title": "beta"}])
        self.assertEqual(self.index.get_documents("alpha"), [])
        self.assertEqual(self.index.get_documents("beta"), [2])
        self.assertEqual(self.index._get_avg_doc_length(), 1)

    def test_empty_build_and_limits(self):
        self.index.build([])
        self.assertEqual(self.index.bm25_search("alpha"), [])
        self.assertEqual(self.index.bm25_search("alpha", 0), [])
        with self.assertRaises(ValueError):
            self.index.bm25_search("alpha", -1)

    def test_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(
                inverted_index,
                CACHE_DIR=root,
                INDEX_PATH=root / "index.pkl",
                DOCMAP_PATH=root / "docs.pkl",
                TF_PATH=root / "tf.pkl",
                DOC_LENGTH_PATH=root / "lengths.pkl",
            ):
                self.index.build([{"id": 1, "title": "alpha"}])
                self.index.save()
                loaded = inverted_index.InvertedIndex()
                loaded.load()
                self.assertEqual(
                    loaded.bm25_search("alpha"), self.index.bm25_search("alpha")
                )


class HybridTests(unittest.TestCase):
    def setUp(self):
        self.hybrid = HybridSearch.__new__(HybridSearch)
        self.hybrid.idx = Mock()
        self.hybrid.semantic_search = Mock()
        self.keyword = {
            "id": 1,
            "title": "keyword",
            "description": "keyword context",
            "score": 3.0,
        }
        self.semantic = {
            "id": 2,
            "title": "semantic",
            "description": "semantic context",
            "score": 0.8,
        }
        self.hybrid.idx.bm25_search.return_value = [self.keyword]
        self.hybrid.semantic_search.search_chunks.return_value = [
            self.semantic,
            {**self.keyword, "score": 0.1},
        ]

    def test_normalization_edge_cases(self):
        self.assertEqual(HybridSearch.normalize([]), [])
        self.assertEqual(HybridSearch.normalize([5, 5]), [1, 1])
        self.assertEqual(HybridSearch.normalize([-2, 0, 2]), [0, 0.5, 1])

    def test_rrf_combines_ranks_and_missing_sources(self):
        results = self.hybrid.rrf_search("query", k=60)
        self.assertEqual(results[0]["id"], 1)
        self.assertAlmostEqual(results[0]["rrf_score"], 1 / 61 + 1 / 62)
        self.assertIsNone(results[1]["bm25_rank"])
        self.assertAlmostEqual(results[1]["rrf_score"], 1 / 61)

    def test_weighted_endpoints(self):
        self.assertEqual(self.hybrid.weighted_search("query", alpha=1)[0]["id"], 1)
        self.assertEqual(self.hybrid.weighted_search("query", alpha=0)[0]["id"], 2)
        self.assertEqual(hybrid_score(1, 0, 0.5), 0.5)

    def test_limits_and_parameters(self):
        self.assertEqual(self.hybrid.rrf_search("query", k=60, limit=0), [])
        self.assertEqual(self.hybrid.weighted_search("query", alpha=0.5, limit=0), [])
        for kwargs in ({"alpha": -0.1}, {"alpha": 1.1}, {"alpha": 0.5, "limit": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.hybrid.weighted_search("query", **kwargs)
        with self.assertRaises(ValueError):
            self.hybrid.rrf_search("query", k=-1)
        self.assertEqual(rrf_score(None), 0)

    def test_missing_or_different_index_rebuilds_supplied_documents(self):
        documents = [{"id": 9, "title": "custom"}]
        for missing in (True, False):
            with (
                self.subTest(missing=missing),
                patch.object(semantic_search, "ChunkedSemanticSearch"),
                patch(
                    "rag_search_engine.core.hybrid_search.InvertedIndex"
                ) as index_class,
            ):
                index = index_class.return_value
                index.docmap = {}
                if missing:
                    index.load.side_effect = FileNotFoundError
                HybridSearch(documents)
                index.build.assert_called_once_with(documents)
                index.save.assert_called_once()


class SemanticTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        paths = patch.multiple(
            semantic_search,
            CACHE_DIR=root,
            EMBEDDINGS_PATH=root / "embeddings.npy",
            CHUNKS_EMBEDDINGS_PATH=root / "chunks.npy",
            CHUNKS_METADATA_PATH=root / "chunks.json",
        )
        paths.start()
        self.addCleanup(paths.stop)
        self.encoder = Mock()
        self.encoder.encode.side_effect = lambda texts, **kwargs: np.ones(
            (len(texts), 3)
        )
        self.encoder.get_sentence_embedding_dimension.return_value = 3
        encoder = patch.object(
            semantic_search, "SentenceTransformer", return_value=self.encoder
        )
        encoder.start()
        self.addCleanup(encoder.stop)
        self.documents = [
            {"id": 1, "title": "first", "description": "A complete story. " * 12}
        ]

    def test_caches_reuse_matching_documents_and_reject_changed_corpora(self):
        changed = [{"id": 2, "title": "second", "description": "Different story."}]
        for search_class, method in (
            (semantic_search.SemanticSearch, "load_or_create_embeddings"),
            (semantic_search.ChunkedSemanticSearch, "load_or_create_chunk_embeddings"),
        ):
            with self.subTest(search_class=search_class.__name__):
                search = search_class()
                load = getattr(search, method)
                load(self.documents)
                calls = self.encoder.encode.call_count
                load(self.documents)
                self.assertEqual(self.encoder.encode.call_count, calls)
                load(changed)
                self.assertEqual(self.encoder.encode.call_count, calls + 1)
                search.model_name = "other-model"
                load(changed)
                self.assertEqual(self.encoder.encode.call_count, calls + 2)

    def test_empty_corpora(self):
        search = semantic_search.SemanticSearch()
        search.load_or_create_embeddings([])
        self.assertEqual(search.search("query", 5), [])
        chunked = semantic_search.ChunkedSemanticSearch()
        chunked.load_or_create_chunk_embeddings([{"id": 1, "title": "empty"}])
        self.assertEqual(chunked.search_chunks("query"), [])

    def test_semantic_results_have_ids_and_full_descriptions(self):
        search = semantic_search.SemanticSearch()
        search.build_embeddings(self.documents)
        self.assertEqual(search.search("query", 1)[0]["id"], 1)
        chunked = semantic_search.ChunkedSemanticSearch()
        chunked.build_chunk_embeddings(self.documents)
        self.assertEqual(
            chunked.search_chunks("query")[0]["description"],
            self.documents[0]["description"],
        )

    def test_cosine_similarity_zero_vectors(self):
        self.assertEqual(semantic_search.cosine_similarity(np.zeros(3), np.ones(3)), 0)
        self.assertAlmostEqual(
            semantic_search.cosine_similarity(np.ones(3), np.ones(3)), 1
        )


class LLMTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {"id": 1, "title": "Title", "description": "Actual retrieved description"}
        ]

    def test_individual_reranker_uses_description(self):
        with (
            patch.object(llm, "invoke_llm", return_value={"response": "8"}) as invoke,
            patch.object(llm.time, "sleep"),
        ):
            ranked = llm.individual_reranker("query", self.results)
        self.assertIn(self.results[0]["description"], invoke.call_args.args[0])
        self.assertEqual(ranked[0]["individual_rerank_score"], 8)

    def test_generators_share_complete_context(self):
        for generator in (
            llm.augmented_generator,
            llm.summarizer,
            llm.citations_generator,
            llm.question_handler,
        ):
            with (
                self.subTest(generator=generator.__name__),
                patch.object(
                    llm, "invoke_llm", return_value={"response": " answer "}
                ) as invoke,
            ):
                self.assertEqual(generator("query", self.results), "answer")
                self.assertIn(self.results[0]["description"], invoke.call_args.args[0])

    def test_invalid_batch_response_falls_back(self):
        for response in ('{"id": 1}', "[{}]", "not JSON"):
            with (
                self.subTest(response=response),
                patch.object(llm, "invoke_llm", return_value={"response": response}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    llm.batch_reranker("query", self.results), self.results
                )

    def test_evaluator_returns_exactly_one_score_per_result(self):
        for response, expected in (
            ("[]", [0]),
            ("[1, 2]", [1]),
            ("[4]", [0]),
            ('{"score": 2}', [0]),
        ):
            with (
                self.subTest(response=response),
                patch.object(llm, "invoke_llm", return_value={"response": response}),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(llm.evaluator("query", self.results), expected)

    def test_optional_completion_fields(self):
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))], usage=None
        )
        with patch.object(llm, "get_client", return_value=client):
            self.assertEqual(
                llm.invoke_llm("prompt"),
                {"response": "", "prompt_tokens": 0, "response_tokens": 0},
            )


class CLITests(unittest.TestCase):
    def test_chunked_cli_prints_description(self):
        with (
            patch("sys.argv", ["rag-semantic", "search_chunked", "query"]),
            patch.object(semantic_search_cli, "load_movies", return_value=[]),
            patch.object(semantic_search_cli, "ChunkedSemanticSearch") as search_class,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            search_class.return_value.search_chunks.return_value = [
                {"title": "Title", "score": 0.9, "description": "Retrieved text"}
            ]
            semantic_search_cli.main()
        self.assertIn("Retrieved text", output.getvalue())

    def test_generation_commands_dispatch_correctly(self):
        results = [{"title": "Title", "description": "Description"}]
        for command, function in (
            ("rag", "augmented_generator"),
            ("summarize", "summarizer"),
            ("citations", "citations_generator"),
            ("question", "question_handler"),
        ):
            with (
                self.subTest(command=command),
                patch("sys.argv", ["rag-generate", command, "query"]),
                patch.object(augmented_generation_cli, "load_movies", return_value=[]),
                patch.object(augmented_generation_cli, "HybridSearch") as search_class,
                patch.object(
                    augmented_generation_cli, function, return_value="Answer"
                ) as generate,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                search_class.return_value.rrf_search.return_value = results
                augmented_generation_cli.main()
                generate.assert_called_once_with("query", results)


class MultimodalTests(unittest.TestCase):
    def test_empty_reranking_does_not_load_model(self):
        with patch.object(cross_encoder, "CrossEncoder") as encoder:
            self.assertEqual(cross_encoder.cross_encoder_reranker("query", []), [])
            encoder.assert_not_called()

    def test_empty_image_search_does_not_encode_empty_corpus(self):
        with patch.object(multimodal_search, "SentenceTransformer") as encoder:
            search = multimodal_search.MultimodalSearch()
            encoder.return_value.encode.assert_not_called()
            self.assertEqual(search.search_with_image("unused.png"), [])


class ConfigurationTests(unittest.TestCase):
    def test_resource_paths_are_absolute(self):
        for path in (
            config.MOVIES,
            config.STOPWORDS,
            config.EVAL_DATASET,
            config.CACHE_DIR,
        ):
            self.assertTrue(path.is_absolute())


if __name__ == "__main__":
    unittest.main()
