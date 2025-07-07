r"""

# Nomenclature

| Prefix | Definition | Examples |
| --- | --- | --- |
| `vn.get_` | Fetch some data | [`vn.get_related_ddl(...)`][vanna.base.base.VannaBase.get_related_ddl] |
| `vn.add_` | Adds something to the retrieval layer | [`vn.add_question_sql(...)`][vanna.base.base.VannaBase.add_question_sql] <br> [`vn.add_ddl(...)`][vanna.base.base.VannaBase.add_ddl] |
| `vn.generate_` | Generates something using AI based on the information in the model | [`vn.generate_sql(...)`][vanna.base.base.VannaBase.generate_sql] <br> [`vn.generate_explanation()`][vanna.base.base.VannaBase.generate_explanation] |
| `vn.run_` | Runs code (SQL) | [`vn.run_sql`][vanna.base.base.VannaBase.run_sql] |
| `vn.remove_` | Removes something from the retrieval layer | [`vn.remove_training_data`][vanna.base.base.VannaBase.remove_training_data] |
| `vn.connect_` | Connects to a database | [`vn.connect_to_snowflake(...)`][vanna.base.base.VannaBase.connect_to_snowflake] |
| `vn.update_` | Updates something | N/A -- unused |
| `vn.set_` | Sets something | N/A -- unused  |

# Open-Source and Extending

Vanna.AI is open-source and extensible. If you'd like to use Vanna without the servers, see an example [here](https://vanna.ai/docs/postgres-ollama-chromadb/).

The following is an example of where various functions are implemented in the codebase when using the default "local" version of Vanna. `vanna.base.VannaBase` is the base class which provides a `vanna.base.VannaBase.ask` and `vanna.base.VannaBase.train` function. Those rely on abstract methods which are implemented in the subclasses `vanna.openai_chat.OpenAI_Chat` and `vanna.chromadb_vector.ChromaDB_VectorStore`. `vanna.openai_chat.OpenAI_Chat` uses the OpenAI API to generate SQL and Plotly code. `vanna.chromadb_vector.ChromaDB_VectorStore` uses ChromaDB to store training data and generate embeddings.

If you want to use Vanna with other LLMs or databases, you can create your own subclass of `vanna.base.VannaBase` and implement the abstract methods.

```mermaid
flowchart
    subgraph VannaBase
        ask
        train
    end

    subgraph OpenAI_Chat
        get_sql_prompt
        submit_prompt
        generate_question
        generate_plotly_code
    end

    subgraph ChromaDB_VectorStore
        generate_embedding
        add_question_sql
        add_ddl
        add_documentation
        get_similar_question_sql
        get_related_ddl
        get_related_documentation
    end
```

"""

import json
import os
import re
import sqlite3
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests
import sqlparse

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path
from langchain_core.documents import Document
from sqlalchemy import create_engine, text


class VannaBase(ABC):
    def __init__(self, config=None):
        if config is None:
            config = {}

        self.config = config
        self.run_sql_is_set = False
        self.static_documentation = config.get("static_documentation", "")
        self.dialect = self.config.get("dialect", "SQL")
        self.language = self.config.get("language", None)
        self.max_tokens = self.config.get("max_tokens", 20000)

    def log(self, message: str, title: str = "Info"):
        print(f"{title}: {message}")

    def _response_language(self) -> str:
        return "Respond in the Catalan language."


    def generate_sql(self, question: str, allow_llm_to_see_data=False, **kwargs) -> str:
        """
        Example:
        ```python
        vn.generate_sql("What are the top 10 customers by sales?")
        ```

        Uses the LLM to generate a SQL query that answers a question. It runs the following methods:

        - [`get_similar_question_sql`][vanna.base.base.VannaBase.get_similar_question_sql]

        - [`get_related_ddl`][vanna.base.base.VannaBase.get_related_ddl]

        - [`get_related_documentation`][vanna.base.base.VannaBase.get_related_documentation]

        - [`get_sql_prompt`][vanna.base.base.VannaBase.get_sql_prompt]

        - [`submit_prompt`][vanna.base.base.VannaBase.submit_prompt]


        Args:
            question (str): The question to generate a SQL query for.
            allow_llm_to_see_data (bool): Whether to allow the LLM to see the data (for the purposes of introspecting the data to generate the final SQL).

        Returns:
            str: The SQL query that answers the question.
        """
    #  Filtra kwargs per evitar passar 'error_feedback' on no toca
        safe_kwargs = {k: v for k, v in kwargs.items() if k != "error_feedback"}

        if self.config is not None:
            initial_prompt = self.config.get("initial_prompt", None)
        else:
            initial_prompt = None

        question_sql_list = self.get_similar_question_sql(question, **safe_kwargs)
        ddl_list = self.get_related_ddl(question, **safe_kwargs)
        doc_list = self.get_related_documentation(question, **safe_kwargs)

        prompt = self.get_sql_prompt(
            initial_prompt=initial_prompt,
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **safe_kwargs,  # 
        )
        self.log(title="SQL Prompt", message=prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        self.log(title="LLM Response", message=llm_response)

        if 'intermediate_sql' in llm_response:
            if not allow_llm_to_see_data:
                return "The LLM is not allowed to see the data in your database. Your question requires database introspection to generate the necessary SQL. Please set allow_llm_to_see_data=True to enable this."

            if allow_llm_to_see_data:
                intermediate_sql = self.extract_sql(llm_response)

                try:
                    self.log(title="Running Intermediate SQL", message=intermediate_sql)
                    df = self.run_sql(intermediate_sql)

                    prompt = self.get_sql_prompt(
                        initial_prompt=initial_prompt,
                        question=question,
                        question_sql_list=question_sql_list,
                        ddl_list=ddl_list,
                        doc_list=doc_list+[f"The following is a pandas DataFrame with the results of the intermediate SQL query {intermediate_sql}: \n" + df.to_markdown()],
                        **kwargs,
                    )
                    self.log(title="Final SQL Prompt", message=prompt)
                    llm_response = self.submit_prompt(prompt, **kwargs)
                    self.log(title="LLM Response", message=llm_response)
                except Exception as e:
                    return f"Error running intermediate SQL: {e}"


        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        """
        Example:
        ```python
        vn.extract_sql("Here's the SQL query in a code block: ```sql\nSELECT * FROM customers\n```")
        ```

        Extracts the SQL query from the LLM response. This is useful in case the LLM response contains other information besides the SQL query.
        Override this function if your LLM responses need custom extraction logic.

        Args:
            llm_response (str): The LLM response.

        Returns:
            str: The extracted SQL query.
        """

        # If the llm_response contains a CTE (with clause), extract the last sql between WITH and ;
        sqls = re.findall(r"\bWITH\b .*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response is not markdown formatted, extract last sql by finding select and ; in the response
        sqls = re.findall(r"SELECT.*?;", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        # If the llm_response contains a markdown code block, with or without the sql tag, extract the last sql from it
        sqls = re.findall(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        sqls = re.findall(r"```(.*)```", llm_response, re.DOTALL)
        if sqls:
            sql = sqls[-1]
            self.log(title="Extracted SQL", message=f"{sql}")
            return sql

        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        """
        Example:
        ```python
        vn.is_sql_valid("SELECT * FROM customers")
        ```
        Checks if the SQL query is valid. This is usually used to check if we should run the SQL query or not.
        By default it checks if the SQL query is a SELECT statement. You can override this method to enable running other types of SQL queries.

        Args:
            sql (str): The SQL query to check.

        Returns:
            bool: True if the SQL query is valid, False otherwise.
        """

        parsed = sqlparse.parse(sql)

        for statement in parsed:
            if statement.get_type() == 'SELECT':
                return True

        return False

    def should_generate_chart(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False

        numeric_df = df.select_dtypes(include=["number"])

        # Retorna True si hi ha almenys una columna numèrica amb alguna dada
        return not numeric_df.empty



    def generate_rewritten_question(
        self, last_question: str, last_sql: str, new_input: str, df: pd.DataFrame, **kwargs
    ) -> str:
        """
        Reescriu o actualitza una pregunta a partir d’un nou input que pot ser una aclariment, 
        una correcció o un seguiment, utilitzant la pregunta original, la SQL i una mostra del resultat (DataFrame).
        
        Args:
            last_question (str): La pregunta original de l’usuari.
            last_sql (str): La consulta SQL que es va generar.
            new_input (str): El nou input de l’usuari (comentari, aclariment, etc.).
            df (pd.DataFrame): Resultat de la consulta SQL.
            **kwargs: Arguments opcionals.
        
        Returns:
            str: Una nova pregunta reformulada que reflecteixi la intenció actualitzada de l’usuari.
        """
        if not last_question and not last_sql:
            return new_input

        # Excloure columnes amb valors massa llargs (com 'geom' o WKB)
        max_len = 200
        filtered_df = df[[col for col in df.columns if df[col].astype(str).map(len).max() < max_len]].copy()

        # Previsualitzar només les primeres 5 files i 5 columnes
        df_preview = (
            filtered_df.iloc[:5, :5].to_markdown(index=False)
            if not filtered_df.empty else "No hi ha dades per mostrar."
        )

        prompt = [
            self.system_message(
                "Ets un assistent de dades que ajuda a l’usuari a interactuar amb una base de dades. "
                "L’usuari ha fet una pregunta que ha generat una consulta SQL i un resultat. Ara proporciona un nou input "
                "que pot ser una aclariment o correcció. Reformula aquest input com una nova pregunta clara i completa. "
                "La nova pregunta ha de poder-se convertir en una nova consulta SQL. Respon només amb la pregunta reformulada, sense explicacions."
            ),
            self.user_message(
                f"Pregunta anterior: {last_question}\n"
                f"Consulta SQL anterior:\n{last_sql}\n"
                f"Resultat (mostra parcial):\n{df_preview}\n"
                f"Nou input: {new_input}"
            )
        ]   

        return self.submit_prompt(prompt=prompt, **kwargs)


    def generate_rewritten_plotly(self, last_question: str, last_plotly_code: str, new_input: str, df: pd.DataFrame, **kwargs) -> str:
        """
        Reescriu o modifica una instrucció per generar una gràfica Plotly, tenint en compte el codi anterior,
        la pregunta inicial, el nou comentari de l’usuari i el DataFrame de dades utilitzat.

        Args:
            last_question (str): La pregunta original que va generar el gràfic.
            last_plotly_code (str): El codi Plotly anterior.
            new_input (str): Nou comentari de l’usuari (correcció o canvi al gràfic).
            df (pd.DataFrame): El DataFrame que es va usar per generar la gràfica.
            **kwargs: Arguments opcionals.

        Returns:
            str: Una nova instrucció reformulada per generar una gràfica millorada o corregida.
        """
        if not last_question and not last_plotly_code:
            return new_input

        prompt = [
            self.system_message(
                "Ets un assistent que ajuda a generar i modificar gràfiques amb Plotly. "
                "A continuació tens una pregunta de l’usuari, el codi Plotly que es va generar, "
                "un DataFrame de dades, i un nou comentari de l’usuari. Reformula aquest comentari com una nova instrucció clara i completa "
                "per generar una gràfica que reflecteixi la seva intenció actual. No afegeixis explicacions."
            ),
            self.user_message(
                f"Pregunta inicial: {last_question}\n"
                f"Codi Plotly anterior:\n{last_plotly_code}\n"
                f"DataFrame:\n{df.head(10).to_markdown(index=False)}\n"
                f"Comentari de l’usuari: {new_input}"
            )
        ]

        return self.submit_prompt(prompt=prompt, **kwargs)
    

    def generate_followup_questions(
        self, question: str, sql: str, df: pd.DataFrame, n_questions: int = 5, **kwargs
    ) -> list:
        """
        Genera una llista de preguntes de seguiment, assegurant que siguin 5, úniques i realment preguntes.

        Args:
            question (str): Pregunta original.
            sql (str): Consulta SQL generada.
            df (pd.DataFrame): Resultats de la consulta.
            n_questions (int): Número de preguntes de seguiment desitjades.

        Returns:
            list: Llista amb 5 preguntes úniques i vàlides.
        """

            # Filtrar columnes amb valors massa llargs (per exemple 'geom')
        max_len = 200
        filtered_df = df[[col for col in df.columns if df[col].astype(str).map(len).max() < max_len]].copy()

        # Mostrar només les primeres 5 files i màxim 5 columnes
        df_preview = (
            filtered_df.iloc[:5, :5].to_markdown(index=False)
            if not filtered_df.empty else "No hi ha dades per mostrar."
        )
        message_log = [
            self.system_message(
                f"Ets un assistent de dades. L’usuari ha preguntat: '{question}'\n\n"
                f"La consulta SQL generada és:\n{sql}\n\n"
                f"El següent és un DataFrame amb els resultats:\n{df_preview}\n\n"
            ),
            self.user_message(
                f"Genera una llista de {n_questions} preguntes de seguiment relacionades amb aquesta consulta i resultats. "
                f"Cada línia ha de ser una pregunta clara, amb resposta possible via SQL. Evita instruccions, codi o frases que no siguin preguntes."
                f"no dejes mucho espacie entre una lina y la siguiente"
            )
        ]

        llm_response = self.submit_prompt(message_log, **kwargs)

        # Dividir en línies
        lines = llm_response.strip().split("\n")

        # Netejar numeració i espais
        cleaned = [re.sub(r"^\d+\.\s*", "", line).strip() for line in lines]

        # Filtrar només preguntes (acaben amb '?')
        questions_only = [q for q in cleaned if q.endswith("?")]

        # Eliminar duplicats mantenint ordre
        seen = set()
        unique_questions = []
        for q in questions_only:
            if q not in seen:
                unique_questions.append(q)
                seen.add(q)

        # Retornar exactament n preguntes
        return unique_questions[:n_questions]



    def generate_questions(self, **kwargs) -> List[str]:
        """
        **Exemple:**
        ```python
        vn.generate_questions()
        ```
    
        Genera una llista de preguntes que pots fer al teu assistent Vanna.
        Es limita a un màxim de 5 preguntes.
        """
        preguntes_sql = self.get_similar_question_sql(question="", **kwargs)
    
        return [q["question"] for q in preguntes_sql[:5]]

    def generate_summary(self, question: str, df: pd.DataFrame, **kwargs) -> str:
        """
        **Example:**
        ```python
        vn.generate_summary("What are the top 10 customers by sales?", df)
        ```

        Generate a summary of the results of a SQL query.

        Args:
            question (str): The question that was asked.
            df (pd.DataFrame): The results of the SQL query.

        Returns:
            str: The summary of the results of the SQL query.
        """
        """         # Excluir columnas problemáticas explícitas
            exclude_columns = {"geom", "geometry", "lat", "lon", "wkb", "x", "y"}
            max_len = 200

            filtered_df = df[
                [col for col in df.columns
                if col.lower() not in exclude_columns
                and df[col].astype(str).map(len).max() < max_len]
            ].copy()
    """
        df_preview = (
            df.head(1).to_markdown(index=False)
            if not df.empty else "No hi ha dades per mostrar."
        )
        
        message_log = [
            self.system_message(
                f"You are a helpful data assistant. The user asked the question: '{question}'\n\nThe following is a pandas DataFrame with the results of the query: \n{df_preview}\n\n"
            ),
            self.user_message(
                "Briefly summarize the data based on the question that was asked. Do not respond with any additional explanation beyond the summary." +
                self._response_language()
            ),
        ]

        summary = self.submit_prompt(message_log, **kwargs)

        return summary

    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        """
        This method is used to get similar questions and their corresponding SQL statements.

        Args:
            question (str): The question to get similar questions and their corresponding SQL statements for.

        Returns:
            list: A list of unique similar questions and their corresponding SQL statements.
        """
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        """
        This method is used to get related DDL statements to a question.

        Args:
            question (str): The question to get related DDL statements for.

        Returns:
            list: A list of related DDL statements.
        """
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        """
        This method is used to get related documentation to a question.

        Args:
            question (str): The question to get related documentation for.

        Returns:
            list: A list of related documentation.
        """
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        """
        This method is used to add a question and its corresponding SQL query to the training data.

        Args:
            question (str): The question to add.
            sql (str): The SQL query to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        """
        This method is used to add a DDL statement to the training data.

        Args:
            ddl (str): The DDL statement to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        """
        This method is used to add documentation to the training data.

        Args:
            documentation (str): The documentation to add.

        Returns:
            str: The ID of the training data that was added.
        """
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        """
        Example:
        ```python
        vn.get_training_data()
        ```

        This method is used to get all the training data from the retrieval layer.

        Returns:
            pd.DataFrame: The training data.
        """
        pass

    @abstractmethod
    def remove_training_data(self, id: str, **kwargs) -> bool:
        """
        Example:
        ```python
        vn.remove_training_data(id="123-ddl")
        ```

        This method is used to remove training data from the retrieval layer.

        Args:
            id (str): The ID of the training data to remove.

        Returns:
            bool: True if the training data was removed, False otherwise.
        """
        pass

    # ----------------- Use Any Language Model API ----------------- #

    @abstractmethod
    def system_message(self, message: str) -> any:
        pass

    @abstractmethod
    def user_message(self, message: str) -> any:
        pass

    @abstractmethod
    def assistant_message(self, message: str) -> any:
        pass

    def str_to_approx_token_count(self, string: str) -> int:
        return len(string) / 4

    def add_ddl_to_prompt(
        self, initial_prompt: str, ddl_list: list[str], max_tokens: int = 14000
    ) -> str:
        if len(ddl_list) > 0:
            initial_prompt += "\n===Tables \n"

            for ddl in ddl_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(ddl)
                    < max_tokens
                ):
                    initial_prompt += f"{ddl}\n\n"

        return initial_prompt

    def add_documentation_to_prompt(
        self,
        initial_prompt: str,
        documentation_list: list[str],
        max_tokens: int = 14000,
    ) -> str:
        if len(documentation_list) > 0:
            initial_prompt += "\n===Additional Context \n\n"

            for documentation in documentation_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(documentation)
                    < max_tokens
                ):
                    initial_prompt += f"{documentation}\n\n"

        return initial_prompt

    def add_sql_to_prompt(
        self, initial_prompt: str, sql_list: list[str], max_tokens: int = 16000
    ) -> str:
        if len(sql_list) > 0:
            initial_prompt += "\n===Question-SQL Pairs\n\n"
            pair_text = f"{question['question']}\n{question['sql']}\n\n"

            for question in sql_list:
                if (
                    self.str_to_approx_token_count(initial_prompt)
                    + self.str_to_approx_token_count(pair_text)
                    < max_tokens
                ):
                    initial_prompt += f"{question['question']}\n{question['sql']}\n\n"

        return initial_prompt
    '''
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

            Example:
            ```python
            vn.get_sql_prompt(
                question="Quin és el número de descàrregues totals durant l’any 2024?",
                question_sql_list=[{"question": "Quin és el nombre de descàrregues totals?", "sql": "SELECT COUNT(*) FROM public.descargas;"}],
                ddl_list=["CREATE TABLE public.descargas (id INT, idfull VARCHAR, fechadescarga TIMESTAMP, ...);"],
                doc_list=["La taula descargas conté informació sobre les descàrregues d'arxius cartogràfics."]
            )
            ```

            Args:
                initial_prompt (str): The base instruction for the system message. If None, a domain-specific one will be generated.
                question (str): The user question to generate SQL for.
                question_sql_list (list): A list of previous question-SQL examples to guide the LLM.
                ddl_list (list): List of DDL statements (table schemas) relevant to the question.
                doc_list (list): List of documentation snippets related to the tables or domain.

            Returns:
                list: A message_log (prompt) ready to be sent to the LLM to generate a SQL query.
            """

            # prompt inicial
            if initial_prompt is None:
                initial_prompt = (
                    "Ets un expert en PostgreSQL en bases de dades de cartografia municipal i estadística d'ús d'un geoportal."
                    "Si us plau, ajuda a generar una consulta SQL per respondre la pregunta. La teva resposta ha d’estar BASADA ÚNICAMENT en el context proporcionat i ha de seguir les directrius de resposta i les instruccions de format. "
                    "No pots utilitzar coneixement extern. "
                    "Genera una consulta PostgreSQL correcta basada exclusivament en aquest context.\n"
                    "No facis servir les taules 'langchain_pg_embedding' ni 'langchain_pg_collection' ni 'spatial_ref_sys' ni 'stored_charts', ja que no contenen informació rellevant per a l'analítica del geoportal.\n"
                )

            # 1. Primero agregar static_documentation si existe
            if self.static_documentation != "":
                doc_list = [self.static_documentation] + doc_list  # Añadimos static_documentation al principio

            # 2. Agregar documentación
            initial_prompt = self.add_documentation_to_prompt(
                initial_prompt, doc_list, max_tokens=self.max_tokens
            )

            # 3. Agregar DDL (estructura de tablas)
            initial_prompt = self.add_ddl_to_prompt(
                initial_prompt, ddl_list, max_tokens=self.max_tokens
            )

        
            # Si se pasa un error anterior, se añade al contexto del prompt
            if "error_feedback" in kwargs:
                initial_prompt += f"\n===Error anterior detectat:\n{kwargs['error_feedback']}\n"


            # 5. Agregar instrucciones claras la satatic_docu respuesta
            initial_prompt += (
                "===Directrius de resposta\n"
                "1. Si el context proporcionat és suficient, genera una consulta SQL sense cap explicació.\n"
                "2. Si el context és gairebé suficient però falta una cadena específica, genera una consulta SQL intermèdia comentada com 'intermediate_sql'.\n"
                "3. Assegura't que les funcions SQL com ROUND(...) tanquin correctament els parèntesis i que l’ús de AS sigui sintàcticament correcte.\n"
                "4. Si el context no és suficient, indica-ho explícitament.\n"
                "5. Fes servir les taules més rellevants.\n"
                "6. Si la pregunta ja té resposta, repeteix la resposta exacta.\n"
                f"7. Assegura que la sortida SQL sigui compatible amb {self.dialect}, executable i sense errors de sintaxi.\n"
                "8. Només pots respondre generant una consulta SQL o indicant explícitament que no pots generar-la. No pots escriure missatges de conversa, salutacions o comentaris personals.\n"
            )

            # 6. Construir el message_log
            message_log = [self.system_message(initial_prompt)]

            # Añadir ejemplos anteriores si existen
            for example in question_sql_list:
                if not question_sql_list:
                    print("no hay ejemplos previos")
                if example is not None and "question" in example and "sql" in example:
                    message_log.append(self.user_message(example["question"]))
                    message_log.append(self.assistant_message(example["sql"]))

            # Finalmente añadir la nueva pregunta del usuario
            message_log.append(self.user_message(question))

            return message_log
    '''
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
        Versión genérica del generador de prompt para generar SQL.
        Puede ser extendida por subclases como ModeloAMB.
        """
        if initial_prompt is None:
            initial_prompt = (
                "You are a SQL expert. Given a question and some documentation, "
                "generate a correct SQL query in PostgreSQL dialect."
            )

        # Añadir documentación (si existe)
        if doc_list:
            initial_prompt += "\n\n=== Documentation ===\n" + "\n".join(doc_list)

        # Añadir DDL (estructura de tablas)
        if ddl_list:
            initial_prompt += "\n\n=== Table Schemas (DDL) ===\n" + "\n".join(ddl_list)

        # Añadir directrices básicas
        initial_prompt += (
            "\n\n=== Instructions ===\n"
            "1. Only return a valid SQL query.\n"
            "2. Do not explain the query.\n"
            "3. Use only the tables and fields described in the documentation and DDL.\n"
        )

        # Armar el prompt como message_log estilo chat
        message_log = [self.system_message(initial_prompt)]

        for example in question_sql_list:
            if example is not None and "question" in example and "sql" in example:
                message_log.append(self.user_message(example["question"]))
                message_log.append(self.assistant_message(example["sql"]))

        message_log.append(self.user_message(question))

        return message_log

    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ) -> list:
        initial_prompt = f"The user initially asked the question: '{question}': \n\n"

        initial_prompt = self.add_ddl_to_prompt(
            initial_prompt, ddl_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_documentation_to_prompt(
            initial_prompt, doc_list, max_tokens=self.max_tokens
        )

        initial_prompt = self.add_sql_to_prompt(
            initial_prompt, question_sql_list, max_tokens=self.max_tokens
        )

        message_log = [self.system_message(initial_prompt)]
        message_log.append(
            self.user_message(
                "Generate a list of unique followup questions that the user might ask about this data. Respond with a list of questions, one per line. Do not answer with any explanations -- just the unique questions."
            )
        )

        return message_log

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        """
        Example:
        ```python
        vn.submit_prompt(
            [
                vn.system_message("The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."),
                vn.user_message("What are the top 10 customers by sales?"),
            ]
        )
        ```

        This method is used to submit a prompt to the LLM.

        Args:
            prompt (any): The prompt to submit to the LLM.

        Returns:
            str: The response from the LLM.
        """
        pass

    def generate_question(self, sql: str, **kwargs) -> str:
        response = self.submit_prompt(
            [
                self.system_message(
                    "The user will give you SQL and you will try to guess what the business question this query is answering. Return just the question without any additional explanation. Do not reference the table name in the question."
                ),
                self.user_message(sql),
            ],
            **kwargs,
        )

        return response

    def _extract_python_code(self, markdown_string: str) -> str:
        # Strip whitespace to avoid indentation errors in LLM-generated code
        markdown_string = markdown_string.strip()

        # Regex pattern to match Python code blocks
        pattern = r"```[\w\s]*python\n([\s\S]*?)```|```([\s\S]*?)```"

        # Find all matches in the markdown string
        matches = re.findall(pattern, markdown_string, re.IGNORECASE)

        # Extract the Python code from the matches
        python_code = []
        for match in matches:
            python = match[0] if match[0] else match[1]
            python_code.append(python.strip())

        if len(python_code) == 0:
            return markdown_string

        return python_code[0]

    def _sanitize_plotly_code(self, raw_plotly_code: str) -> str:
        # Remove the fig.show() statement from the plotly code
        plotly_code = raw_plotly_code.replace("wkb_loads", "shapely.wkb.loads")
        plotly_code = plotly_code.replace("fig.show()", "")

            # Eliminar llamadas directas a st_folium(...) que podrían duplicar el mapa
        lines = plotly_code.splitlines()
        lines = [line for line in lines if not line.strip().startswith("st_folium(")]
        sanitized_code = "\n".join(lines)

        return sanitized_code

    """  def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"The following is a pandas DataFrame that contains the results of the query that answers the question the user asked: '{question}'"
        else:
            system_msg = "The following is a pandas DataFrame "

        if sql is not None:
            system_msg += f"\n\nThe DataFrame was produced using this query: {sql}\n\n"

        system_msg += f"The following is information about the resulting pandas DataFrame 'df': \n{df_metadata}"

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Can you generate the Python plotly code to chart the results of the dataframe? Assume the data is in a pandas dataframe called 'df'. If there is only one value in the dataframe, use an Indicator. Respond with only Python code. Do not answer with any explanations -- just the code."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))
         """

    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, chart_type: str = None, **kwargs
    ) -> str:
        if question is not None:
            system_msg = f"Tens un DataFrame de pandas que conté els resultats de la consulta SQL relacionada amb la pregunta: '{question}'"
        else:
            system_msg = "Tens un DataFrame de pandas amb resultats analítics."

        if sql is not None:
            system_msg += f"\n\nEl DataFrame s’ha generat a partir de la següent consulta SQL:\n{sql}\n"

        system_msg += (
            f"\nAquest DataFrame s’anomena `df` i conté les següents columnes i tipus:\n{df_metadata}\n\n"
            f"Genera codi Python amb Plotly per visualitzar les dades. Els requisits són:\n"
            f"- Si només hi ha un valor, utilitza un gràfic Indicator.\n"
            f"- Les etiquetes (títols, eixos, llegendes) han d’estar en català.\n"
            f"- Els noms de les columnes han d’estar formatats amb espais (substitueix '_' per ' ').\n"
            f"- Utilitza `color=` només si hi ha una columna categòrica rellevant amb poques categories que aporti valor visual al gràfic (com el nom del producte, el gènere, etc).\n"
            f"- Si només hi ha una única sèrie (una sola variable dependent sense desglossament en categories), no afegeixis `color=`. Utilitza un sol color uniforme.\n"
            f"- No utilitzis `color=` si el gràfic mostra una seqüència temporal o contínua (com mesos o dies) sense desglossament per categories.\n"
            f"- En els gràfics de tipus pastís (pie chart), utilitza `color=` per distingir les categories.\n"
            f"- Els gràfics han de tenir un estil net, seriós i adequat per informes institucionals.\n"



        )

        if chart_type:
            system_msg += (f"\n- El gràfic ha de ser de tipus: **{chart_type}**."
            "no en generis cap altre tipus. "
            "Ignora qualsevol altre format encara que el DataFrame pugui suggerir-ne un diferent.")

        system_msg += "\n- Torna només codi Python, sense cap explicació."

        message_log = [
            self.system_message(system_msg),
            self.user_message(
                "Assumeix que ja existeix un DataFrame anomenat `df` amb les dades. "
                "No creïs un DataFrame nou ni afegeixis dades fictícies. "
                "Mostra el codi en Python amb Plotly per visualitzar `df`. "
                "Torna només codi Python, sense cap explicació."
            ),
        ]

        plotly_code = self.submit_prompt(message_log, kwargs=kwargs)

        return self._sanitize_plotly_code(self._extract_python_code(plotly_code))

    def get_all(self):
        """
        Devuelve todos los documentos almacenados en la colección actual (sql, ddl o documentation).
        Esta función solo funciona si se llama desde self.sql_collection, self.ddl_collection o self.documentation_collection.
        """
        engine = create_engine(self.connection_string)

        query = text("""
            SELECT document, cmetadata
            FROM langchain_pg_embedding
            WHERE cmetadata->>'collection_name' = :collection
        """)

        # Inferir el nombre de la colección activa (sql, ddl o documentation)
        if hasattr(self, "collection_name"):
            collection_name = self.collection_name
        else:
            raise AttributeError("Falta 'collection_name' en el contexto de la colección.")

        with engine.connect() as conn:
            results = conn.execute(query, {"collection": collection_name}).fetchall()

        return [
            Document(page_content=row["document"], metadata=row["cmetadata"])
            for row in results
        ]





    # ----------------- Connect to Any Database to run the Generated SQL ----------------- #


    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
        **kwargs
    ):

        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][vanna.base.base.VannaBase.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install vanna[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
                **kwargs
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def connect_to_db():
            return psycopg2.connect(host=host, dbname=dbname,user=user, password=password, port=port, **kwargs)


        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            conn = None
            try:
                conn = connect_to_db()  # Initial connection attempt
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.InterfaceError as e:
                # Attempt to reconnect and retry the operation
                if conn:
                    conn.close()  # Ensure any existing connection is closed
                conn = connect_to_db()
                cs = conn.cursor()
                cs.execute(sql)
                results = cs.fetchall()

                # Create a pandas dataframe from the results
                df = pd.DataFrame(results, columns=[desc[0] for desc in cs.description])
                return df

            except psycopg2.Error as e:
                if conn:
                    conn.rollback()
                    raise ValidationError(e)

            except Exception as e:
                        conn.rollback()
                        raise e

        self.dialect = "PostgreSQL"
        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres

