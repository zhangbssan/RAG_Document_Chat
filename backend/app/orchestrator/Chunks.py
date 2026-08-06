import json
import os
from langchain_community.document_loaders import PyPDFLoader,Docx2txtLoader, UnstructuredMarkdownLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
import config
from pymilvus import MilvusClient, DataType
from pymilvus.model.sparse import BM25EmbeddingFunction
import time



class Chunks_Embedding(object):
    def __init__(self):
        self.loader = {
            ".pdf": PyPDFLoader,
            ".txt": Docx2txtLoader,
            ".docx": Docx2txtLoader,
            ".md": UnstructuredMarkdownLoader,
            ".csv": CSVLoader,
            ".json": self.handle_json,

        }
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size =config.CHUNK_SIZE , 
                                                           chunk_overlap = config.CHUNK_OVERLAP,
                                                           separators=["\n\n", "\n", " ", ""],
                                                           length_function = len, 
                                                           add_start_index = True)
        self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL,
                                           base_url="http://localhost:11434"  
                                           )
    def get_file(self, filename):
        file_extension = os.path.splitext(filename)[-1]
        loader = self.loader.get(file_extension,None)
        if loader:
            if file_extension ==".json":
                return loader(filename)
            else:
                loader_info = loader(filename).load()
                return loader_info
        else:
            return None
        
    def handle_json(self, filename):
        with open(filename, "r") as f:
            data = f.read()
        return data
    def is_json(self, data):
        try:
            json.loads(data)
            return True
        except:
            return False
    
    def split_text(self, filename):
        load_info = self.get_file(filename)
        if load_info:
            if self.is_json(load_info):
                DATA = json.loads(load_info)
                results = []
                for item in DATA:
                    name = item.get("product_name","")
                    desc = item.get("description", "")
                    page_content = f"{name}. Funktion and Anwendung: {desc}"

                    metadata = {
                        "item_id": item.get("item_id",""),
                        "stock" : item.get("stock",0),
                        "location": item.get("location","")
                    }
                    doc = Document(page_content=page_content, metadata= metadata)
                    results.append(doc)
                self.chunks = results
                print(f"✅ JSON-Verarbeitung abgeschlossen! Erfolgreich {len(DATA)} Produkte in {len(self.chunks)} unabhängige Chunks konvertiert.")
            else:
                self.chunks = self.text_splitter.split_documents(load_info)
                print(f"    => Chunking erfolgreich: Insgesamt {len(self.chunks)} Textabschnitte generiert.")
            return self.chunks
        else:
            raise "not support"
        
    def vector_storage(self,chunks):
        start_time = time.time()
        texts = [doc.page_content for doc in chunks]
       
        # 1. Sparse Vektoren generieren (BM25)
        bm25_ef = BM25EmbeddingFunction()
        bm25_ef.fit(texts)
        sparse_vectors = bm25_ef.encode_documents(texts)

        # 2. Dense Vektoren generieren (Semantische Embeddings)
        dense_vectors = self.embeddings.embed_documents(texts)

        # 3. Verbindung zur Datenbank herstellen
        client = MilvusClient(uri=f"http://{config.MILVUS_HOST}:{config.MILVUS_PORT}")
        collection_name = config.COLLECTION_NAME

        if client.has_collection(collection_name):
            client.drop_collection(collection_name)

        # 4. Datenbankschema aufbauen (mit dynamischen Feldern)
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name = "id", datatype=DataType.INT64, is_primary=True)
        schema.add_field(field_name="page_content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=768) 
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        client.create_collection(collection_name=collection_name, schema=schema)
       
        # 5. Suchindizes erstellen
        index_params = client.prepare_index_params()
        index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="IP")
        index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
        client.create_index(collection_name, index_params)

        # 6. Sparse-Matrix in Dictionaries umwandeln (Speicheroptimierung)
        data_to_insert = []
        sparse_coo = sparse_vectors.tocoo()
        sparse_dicts = [{} for _ in range(len(chunks))]
        for r, c, v in zip(sparse_coo.row, sparse_coo.col, sparse_coo.data):
            sparse_dicts[int(r)][int(c)] = float(v)

        # 7. Daten assemblieren und Metadaten bereinigen
        for i, doc in enumerate(chunks):
            clean_metadata = {}
            for key, value in doc.metadata.items():
                new_key = key.replace(".", "_") 
                clean_metadata[new_key] = value
            record = {
                "page_content": doc.page_content,
                "dense_vector": dense_vectors[i],
                "sparse_vector": sparse_dicts[i],
                **clean_metadata  
            }
            data_to_insert.append(record)
            
        client.insert(collection_name=collection_name, data=data_to_insert)
        client.load_collection(collection_name)
        
        end_time = time.time()
        print(f"🎉 Erfolgreich! Dauer: {end_time - start_time:.2f} Sekunden")
        print(f"    => Native Hybrid-Daten wurden erfolgreich in '{config.COLLECTION_NAME}' gespeichert.")
        self.bm25_ef = bm25_ef 
        self.vector_db = client
        
        return client
    
if __name__ == '__main__':
    if not os.path.exists(config.FILE_PATH):
        print(f"❌ Fehler: Datei {config.FILE_PATH} nicht gefunden.")
        print(f"Bitte legen Sie die Datei in den Ordner '{config.DATA_DIR}' und überprüfen Sie den Dateinamen in der config.py.")
    else:
        processor = Chunks_Embedding()
        all_chunks = processor.split_text(config.FILE_PATH)
        print(all_chunks[0])
        vectors=processor.vector_storage(all_chunks)
