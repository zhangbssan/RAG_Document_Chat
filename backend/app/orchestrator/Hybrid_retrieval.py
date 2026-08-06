from Biencoder_dense_retrieval import DenseRetriever
from BM25_sparse_retrieval import SparseRetriever
import Chunks
import config
import os
class HybridSearcher:
    def __init__(self, processor, collection_name=None):
        print("\n🚀 Initialisiere vergleichendes hybrides Retrieval-System...")
        
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.dense_retriever = DenseRetriever(collection_name=self.collection_name)
        self.sparse_retriever = SparseRetriever(processor, collection_name=self.collection_name)

    def search(self, query, top_k=2):
        print(f"\n🔎 [Hybrid] Führe hybride Vergleichssuche aus: '{query}'")
        
        # 1. Dense-Suche ausführen (Fokus auf semantischen Kontext)
        dense_results = self.dense_retriever.search(query, top_k=top_k)
        print(f"   - Dense (Semantik): {len(dense_results)} Treffer")

        # 2. Sparse-Suche ausführen (Strikte Keyword-Übereinstimmung via BM25)
        sparse_results = self.sparse_retriever.search(query, top_k=top_k)
        print(f"   - Sparse (BM25-Keywords): {len(sparse_results)} Treffer")
        
        # 3. Beide Ergebnislisten in einem Dictionary zusammenfassen und zurückgeben!
        combined_results = {
            "dense": dense_results,
            "sparse": sparse_results
        }
        
        return combined_results

if __name__ == '__main__':
    if not os.path.exists(config.FILE_PATH):
        print(f"❌ Fehler: Datei {config.FILE_PATH} nicht gefunden.")
        print(f"Bitte legen Sie die Datei in den Ordner '{config.DATA_DIR}' und überprüfen Sie den Dateinamen in der config.py.")
    else:
      
        processor = Chunks.Chunks_Embedding()
        all_chunks = processor.split_text(config.FILE_PATH)
        processor.vector_storage(all_chunks)

        hybrid_searcher = HybridSearcher(processor)

        test_query = "Ich brauche ein Kupferrohr für das Waschbecken"
        
        print("\n" + "="*60)
        print(f"🗣️  Kundenanfrage : '{test_query}'")
        print("="*60)

        results = hybrid_searcher.search(test_query, top_k=2)

        print("-" * 50)
        for i, res in enumerate(results["dense"]):
            meta = res['metadata']
            print(f"   {i+1}. 🆔 {meta['item_id']} | 📦 Bestand: {meta['stock']} | 🎯 Score: {res['score']:.4f}")
            print(f"      📄 Text: {res['content'][:80]}...")

        print("\n🔑 [Ergebnisse] Sparse Search (BM25):")
        print("   -> Stärke: Findet exakte Fachbegriffe (z.B. 'Kupferrohr')")
        print("-" * 50)
        for i, res in enumerate(results["sparse"]):
            meta = res['metadata']
            print(f"   {i+1}. 🆔 {meta['item_id']} | 📦 Bestand: {meta['stock']} | 🎯 Score: {res['score']:.4f}")
            print(f"      📄 Text: {res['content'][:80]}...")
            
        print("\n✅ Demo erfolgreich abgeschlossen!")
