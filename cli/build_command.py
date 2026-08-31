from inverted_index import InvertedIndex

def build_command() -> None:
    # 1. Instantiate the index
    idx = InvertedIndex()

    # 2. Build the index from movies
    idx.build()

    # 3. Save to disk (cache/index.pkl and cache/docmap.pkl)
    idx.save()

    # 4. Test output required by the test runner
    docs = idx.get_documents("merida")
    print(f"First document for token 'merida' = {docs[0]}")