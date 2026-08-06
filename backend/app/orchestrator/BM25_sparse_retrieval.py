from pymilvus import Collection
import config

class SparseRetriever:
    def __init__(self, processor, collection_name=None):
        
        self.config = config
        self.bm25_ef = processor.bm25_ef
        self.collection_name = collection_name or self.config.COLLECTION_NAME
        
        # 1. Datenbank verbinden und Collection laden
        self.collection = Collection(self.collection_name)
        self.collection.load()
        print(
            f"✅ [Sparse] Erfolgreich mit der Milvus Sparse-Engine verbunden: "
            f"{self.collection_name}"
        )

    def search(self, query_text, top_k=3):
        
        query_matrix = self.bm25_ef.encode_queries([query_text])
        
        coo = query_matrix.tocoo()
        query_sparse = {int(c): float(v) for c, v in zip(coo.col, coo.data)}
        
        # Dynamische Felder abfragen:
        # - immer page_content ausgeben
        # - alle anderen Metadatenfelder automatisch mitnehmen
        # - Vektorfelder und Primärschlüssel für Ausgabe überspringen
        output_fields = ["page_content"]
        metadata_fields = []
        for field in self.collection.schema.fields:
            field_name = field.name
            if field_name in {"id", "page_content", "dense_vector", "sparse_vector"}:
                continue
            output_fields.append(field_name)
            metadata_fields.append(field_name)

        # 2. Suche in der 'sparse_vector' Spalte der Datenbank ausführen (Inner Product)
        results = self.collection.search(
            data=[query_sparse],
            anns_field="sparse_vector", 
            param={"metric_type": "IP"},
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
                    "type": "sparse",
                    "metadata": metadata
                })
        
        return formatted_results