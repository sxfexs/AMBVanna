from ..pgvector import PG_VectorStore
from langchain_core.documents import Document
import ast
from sqlalchemy import create_engine, text

# 🔹 Configuración del vector store
config = {
    "connection_string": "postgresql://postgres.spdwbcfeoefxnlfdhlgi:chatbot2025@aws-0-eu-central-1.pooler.supabase.com:6543/postgres?options=-csearch_path=vector_store"
}

#  Clase personalizada con embeddings en catalán
'''Los embeddings son vectores numéricos (listas de números) que representan el significado de un texto en un espacio matemático.
Es decir: convierten una frase como "quants usuaris són dones?" en una lista de números como [0.12, -0.45, 0.89, ...].
Esto permite que el sistema compare preguntas parecidas aunque estén escritas diferente.
MUY IMPORTANTE, SE CAMBIO EL CODIGO DEL EMBADDING THE VANNABASE YA QUE ESTE NO ESTA ELABORANDO LOS VECTORES EN CATALAN.  
si el modelo de embeddings no entiende cataláN convierte frases en catalán a vectores que no reflejan bien su sentido. Se recuperan ejemplos poco útiles o incluso incorrectos.
Un enbedding que entiende catalan  "intfloat/multilingual-e5-base", en el vannabase usaba all-MiniLM-L6-v2 que no entiende catalán.
'''

class AMB_VectorStore(PG_VectorStore):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.embedding_model = None
        
        
    def generate_embedding(self, data: str, **kwargs) -> list:
        if self.embedding_model is None:
            print("Cargando modelo de embeddings...")
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
        return self.embedding_model.encode(data).tolist()
    
    def system_message(self, message: str) -> any:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> any:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> any:
        return {"role": "assistant", "content": message}

    def submit_prompt(self, prompt, **kwargs) -> str:
        return "This is a placeholder response from the model."
    
    def save_plotly_and_map_code(self, id: str, plotly_code: str, map_code: str) -> bool:
        """
        Guarda el código Plotly y el código del mapa (Folium) en la base de datos.

        Args:
            id (str): ID generado al guardar la pregunta y SQL.
            plotly_code (str): Código Plotly a guardar.
            map_code (str): Código Folium a guardar.

        Returns:
            bool: True si la inserción fue exitosa, False en caso contrario.
        """
        from sqlalchemy import create_engine, text

        try:
            engine = create_engine("postgresql://postgres.spdwbcfeoefxnlfdhlgi:chatbot2025@aws-0-eu-central-1.pooler.supabase.com:6543/postgres")
            query = text("""
                INSERT INTO stored_charts (id, plotly_code, map_code)
                VALUES (:id, :plotly_code, :map_code)
                ON CONFLICT (id) DO UPDATE
                SET plotly_code = EXCLUDED.plotly_code,
                    map_code = EXCLUDED.map_code
            """)
            with engine.connect() as conn:
                conn.execute(query, {
                    "id": id,
                    "plotly_code": plotly_code,
                    "map_code": map_code
                })
                conn.commit()
            return True
        except Exception as e:
            print(f"Error guardando código Plotly y mapa: {e}")
            return False

    def get_all(self, collection_name: str) -> list:
        engine = create_engine(self.connection_string)

        query = text("""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'collection_name' = :collection
        """)

        with engine.connect() as conn:
            results = conn.execute(query, {"collection": collection_name}).fetchall()

        return [
            Document(page_content=row["document"], metadata=row["cmetadata"])
            for row in results
        ]


    def get_similar_question_sql(self, question: str) -> list:
        # Recuperar resultados por similitud
            documents = self.sql_collection.similarity_search(query=question, k=self.n_results)
            results = [ast.literal_eval(document.page_content) for document in documents]

        # Comprobar si ya existe una coincidencia exacta
            for r in results:
                if r["question"].strip() == question.strip():
                    return [r]  # Priorizar exacta

        # Si no está entre los top-k, buscarla en toda la colección (opcional pero más costoso)
            all_docs = self.get_all("sql")# Si tu vectorstore lo permite
             
            for doc in all_docs:
                parsed = ast.literal_eval(doc.page_content)
                if parsed["question"].strip() == question.strip():
                    return [parsed] + results  # Forzarla al frente

            return results

# Instancia del vector store
vector_store = AMB_VectorStore(config)
