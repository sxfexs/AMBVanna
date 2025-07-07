import time
start_time = time.time()
from transformers import AutoTokenizer, AutoModelForCausalLM
print("⏱ Finalizado import transformer:", time.time() - start_time, "segundos")
start_time2 = time.time()
from ..base import VannaBase
print("⏱ Finalizado VannaBase:", time.time() - start_time2, "segundos")
start_time3 = time.time()
from datetime import datetime
from accelerate import infer_auto_device_map
import torch
print("⏱ Finalizado datetime,accelerate y torch:", time.time() - start_time3, "segundos")

class ModeloAMB(VannaBase):
    def __init__(self, config=None):
        super().__init__(config or {})
        self.config = config or {}

        model_name_or_path = self.config.get("model_name_or_path")
        token = self.config.get("token")
        cache_dir = "E:/Chatbot/Models/huggingface"

        print("Cargando tokenizer base...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=token,
            cache_dir=cache_dir,
            legacy=False
        )

        print("Cargando modelo base sin PEFT...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=token,
            cache_dir=cache_dir
        )

        # Despachar el modelo a los dispositivos
        device_map = infer_auto_device_map(
            self.model,
            max_memory={0: "46GiB", 1: "46GiB"},
            no_split_module_classes=["MistralDecoderLayer"]
        )
        from accelerate import dispatch_model
        self.model = dispatch_model(self.model, device_map=device_map)

        print("Modelo base cargado correctamente.")


    
