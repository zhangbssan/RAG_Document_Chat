import ollama
from pymilvus import connections, Collection, utility
import config

class DenseRetriever:
    def __init__(self, collection_name=None):
       
        self.config = config
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        self.collection = None
        self.search_params = {}
        self._connect_milvus()
        self._load_collection_and_params()

    def _connect_milvus(self):
        try:
            connections.connect(alias="default", host=self.config.MILVUS_HOST, port=self.config.MILVUS_PORT)
            print(f"✅ [Dense] Erfolgreich mit Milvus verbunden ({self.config.MILVUS_HOST}:{self.config.MILVUS_PORT})")
        except Exception as e:
            print(f"❌ [Dense] Verbindung zu Milvus fehlgeschlagen: {e}")
            raise

    def _load_collection_and_params(self):
        if not utility.has_collection(self.collection_name):
            raise ValueError(f"❌ [Dense] Collection '{self.collection_name}' existiert nicht!")

        self.collection = Collection(self.collection_name)
        self.collection.load()

        if not self.collection.indexes:
            print("⚠️ [Dense] Warnung: Diese Collection hat keinen Index. Die Suche könnte sehr langsam sein oder fehlschlagen.")
            self.search_params = {"metric_type": "L2", "params": {}}
            return

        index = self.collection.indexes[0]
        index_args = index.params
        metric_type = index_args.get("metric_type", "L2")

        params = {}
        idx_name = index.index_name.upper() if hasattr(index, 'index_name') else "UNKNOWN"

        
        if "HNSW" in idx_name:
            params = {"ef": 50} 
            print(f"✅ [Dense] HNSW-Index erkannt, setze Suchparameter ef=50")
        elif "IVF" in idx_name:
            params = {"nprobe": 10}
            print(f"✅ [Dense] IVF-Index erkannt, setze Suchparameter nprobe=10")
        else:
            params = {"ef": 10}
            print(f"⚠️ [Dense] Unbekannter Indextyp ({idx_name}), verwende Fallback-Parameter")

        # Finale Suchparameter speichern
        self.search_params = {
            "metric_type": metric_type,
            "params": params
        }
    def _get_embedding(self, text):
        response = ollama.embeddings(model=self.config.EMBEDDING_MODEL, prompt=text)
        return response["embedding"]

    def search(self, query_text, top_k=3):
            # 1. Validierung: Prüfen, ob die Datenbankverbindung steht
            if not self.collection:
                raise ValueError("❌ [Dense] Collection wurde nicht geladen!")

            query_vector = self._get_embedding(query_text)

            output_fields = ["page_content"]
            metadata_fields = []
            for field in self.collection.schema.fields:
                field_name = field.name
                if field_name in {"id", "page_content", "dense_vector", "sparse_vector"}:
                    continue
                output_fields.append(field_name)
                metadata_fields.append(field_name)

            # 2. Vektorsuche in Milvus durchführen (ANN - Approximate Nearest Neighbor)
            results = self.collection.search(
                data=[query_vector],
                anns_field="dense_vector",     
                param=self.search_params,
                limit=top_k,
                output_fields=output_fields   
            )

            # 3. Ergebnisse formatieren (Mapping auf einheitliches JSON-Format)
            formatted_results = []
            for hits in results:
                for hit in hits:
                    metadata = {}
                    for key in metadata_fields:
                        value = hit.entity.get(key)
                        if value is not None:
                            metadata[key] = value
                    formatted_results.append({
                        "content": hit.entity.get("page_content"),
                        "score": hit.score, 
                        "type": "dense",     
                        "metadata": metadata
                    })
            
            return formatted_results