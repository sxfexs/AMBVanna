from .modelo_amb.vector_store import AMB_VectorStore
from .modelo_amb.modelo_amb import ModeloAMB

class AmbVanna(ModeloAMB, AMB_VectorStore):
    def __init__(self, config=None):
        ModeloAMB.__init__(self, config=config) # maneja la lógica de generación (prompts, tokenizer, modelo LLM).
        AMB_VectorStore.__init__(self, config=config) # maneja la lógica de recuperación, embeddings y base de dato del contexto 
        
