from inverted_index import InvertedIndex

idx = InvertedIndex()
idx.load()
print("avg_len:", idx._InvertedIndex__get_avg_doc_length())
print("doc 2275 len:", idx.doc_lengths[2275])
print("doc 1907 len:", idx.doc_lengths[1907])