from vanna.modelo_amb.modelo_amb import ModeloAMB
from .vector_store import AMB_VectorStore
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time
from accelerate import infer_auto_device_map
from datetime import datetime
import os

class AmbVannaSQL(ModeloAMB, AMB_VectorStore):
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        # Inicializa ModeloAMB, lo que carga el tokenizer y el modelo base
        super().__init__(config=config)

        # Inicializa el vector store
        AMB_VectorStore.__init__(self, config=config)

        # Cargar configuración PEFT y aplicar pesos sobre self.model
        from peft import PeftModel, PeftConfig
        print("Aplicando pesos PEFT a modelo base...")

        peft_config = PeftConfig.from_pretrained(config["model_name_or_path"])
        self.model = PeftModel.from_pretrained(self.model, config["model_name_or_path"])

        # Mapear en GPU
        from accelerate import dispatch_model, infer_auto_device_map
        device_map = infer_auto_device_map(
            self.model,
            max_memory={0: "46GiB", 1: "46GiB"},
            no_split_module_classes=["MistralDecoderLayer"]
        )
        self.model = dispatch_model(self.model, device_map=device_map)

        print("Modelo PEFT cargado correctamente sobre base.")

    def prompt_de_txt(self, file_path):
        base_dir = os.path.dirname(__file__)
        full_path = os.path.join(base_dir, file_path)
        if not os.path.exists(full_path):
            print(f"❌ Archivo no encontrado: {full_path}")
            return None
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"❌ Error al leer el archivo: {e}")
            return None
    
    def get_sql_prompt(
        self,
        initial_prompt: str,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        """
        Genera un prompt para que el modelo LLM cree consultas SQL, enfocándose específicamente en el contexto del geoportal de cartografía del AMB

        """
        
        # Carga el initial_prompt de txt en la carpeta prompts
        initial_prompt = self.prompt_de_txt('Prompts/initial_prompt.txt')
 
        # Agregar DDL (estructura de tablas)
        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        # Agregar documentación
        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )
   
        # Si se pasa un error anterior, se añade al contexto del prompt
        if "error_feedback" in kwargs:
            initial_prompt += f"\n===Error anterior detectat:\n{kwargs['error_feedback']}\n"
  
        # Construir el message_log
        message_log = [self.system_message(initial_prompt)]

        # Añadir ejemplos anteriores si existen
        for example in question_sql_list:
            if not question_sql_list:
                print("no hay ejemplos previos")
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        # Finalmente añadir la pregunta que esta haciendo el usuario
        message_log.append(self.user_message(question))

        return message_log

    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}

    def extract_sql_query(self, text: str) -> str:
        """
        Extrae la primera consulta SQL válida después de 'SELECT',
        eliminando caracteres no deseados.
        """
        sql = super().extract_sql(text)
        return sql.replace("\\_", "_").replace("\\", "")
    
    def corregir_sql_any(self, sql: str) -> str:
        """
        Corrige el uso incorrecto de 'any' como alias de año.
        """
        # Casos comunes de alias
        sql = sql.replace(" AS any", " AS y")
        sql = sql.replace(" as any", " AS y")
        sql = sql.replace(" BY any", " BY y")
        sql = sql.replace(" by any", " BY y")
    
        # Casos en SELECT, GROUP BY, etc.
        sql = sql.replace("SELECT any", "SELECT y")
        sql = sql.replace(", any", ", y")
        sql = sql.replace(" any,", " y,")
        sql = sql.replace(" any ", " y ")
        sql = sql.replace("any;", " y; ")



        return sql

    
    def generate_sql(self, question: str, **kwargs) -> str:
        # Usa la función base de Vanna
        sql = super().generate_sql(question, **kwargs)

        # Limpiezas comunes
        sql = sql.replace("\\_", "_")
        sql = sql.replace("\\", "")

        # Corrección de errores comunes como 'any' mal usado
        sql = self.corregir_sql_any(sql)

        # Extrae la SQL final (por si viene con bloques adicionales)
        return self.extract_sql_query(sql)

    def submit_prompt(self, prompt, **kwargs) -> str:
        """

    Envía un prompt al modelo de lenguaje y devuelve la resposta generada.
        
        """
       
        if prompt[0]["role"] == "system":
            system_msg = prompt[0]["content"]
            # Pegamos esta instrucción al principio de la primera 'user'
            for i in range(1, len(prompt)):
                if prompt[i]["role"] == "user":
                    prompt[i]["content"] = system_msg + "\n\n" + prompt[i]["content"]
                    break
            prompt = prompt[1:]  # Eliminamos el 'system' ya que ya lo usamos

        #  Mostrem el prompt abans de la generació
        print("=== Prompt tokens decodificats ===")
        for p in prompt:
            print(f"{p['role']}: {p['content']}")

        # Apliquem el format de plantilla segons el model
        full_prompt = self.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            date_string=datetime.today().strftime("%Y-%m-%d")
        )


        # Codifiquem el prompt
        input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.model.device)

        print("\n==== PROMPT REAL AL MODELO (para SQL)====\n", full_prompt)
        print("\n==== LONGITUD (tokens) ====\n", len(input_ids[0]))

        # Generem la resposta
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=3000,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=False,
            temperature=0.2,
            top_p=1.0,
        )

        # Extraiem només la part generada
        response = outputs[0][input_ids.shape[-1]:]
        response = self.tokenizer.decode(response, skip_special_tokens=True)

        self.log(response)

        return response
    
