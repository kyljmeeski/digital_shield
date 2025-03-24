from sentence_transformers import SentenceTransformer

if __name__ == "__main__":
    SEMANTIC_ANALYZER = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    SEMANTIC_ANALYZER.save('paraphrase-multilingual-MiniLM-L12-v2')  # for the first time only -- to save locally
