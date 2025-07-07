from vanna.modelo_amb.modelo_amb import ModeloAMB
from .vector_store import AMB_VectorStore
import torch
from datetime import datetime
import os

class AmbVannaGeneral(ModeloAMB, AMB_VectorStore):
    def __init__(self, config=None):
        super().__init__(config=config)
        AMB_VectorStore.__init__(self, config=config)
        print("Modelo general usa el modelo base tal cual .")
    
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

    def generate_map_code(self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs) -> str:
        """
    Utilitza el LLM per generar codi Python per visualitzar geodades a partir del DataFrame `df` amb Streamlit i Folium,
    seguint patrons específics per a diferents tipus de geometries i consultes geoespacials.
        """

        if question is not None:
            system_msg = (
                f"Tens un DataFrame anomenat `df` que conté els resultats de la consulta SQL relacionada amb la pregunta: '{question}'."
            )
        else:
            system_msg = "Tens un DataFrame anomenat `df` amb resultats geoespacials."

        if sql is not None:
            system_msg += f"\n\nEl DataFrame s’ha generat a partir de la següent consulta SQL:\n{sql}\n"

        # Cargar texto adicional desde un archivo
        system_msg += "\n\n" + self.prompt_de_txt('Prompts/initial_prompt_mapas.txt')


        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Genera només codi Python per visualitzar el DataFrame `df` com a mapa interactiu amb Streamlit i Folium."
            )
        ]

        #plot_code = self.submit_prompt(message_log, kwargs=kwargs)
        plot_code = self.submit_prompt(message_log, **kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plot_code))


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

        print("\n==== PROMPT REAL AL MODELO ====\n", full_prompt)
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
    
    def system_message(self, message: str) -> dict:
        return {"role": "system", "content": message}

    def user_message(self, message: str) -> dict:
        return {"role": "user", "content": message}

    def assistant_message(self, message: str) -> dict:
        return {"role": "assistant", "content": message}
