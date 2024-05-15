import torch
from typing import List, Dict, Union
from jsonformer import Jsonformer
import transformers
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForTokenClassification
import pandas as pd
from src.Utils import get_concat_col_name, get_left_right_concat, get_concat
from preprocessing.Attributes import BASELINE_NAMES


class LlamaWrapper(object):
    """A wrapper for using Llama.
    Args:
            model_id (str): The model ID to load from huggingface.
    """
    def __init__(
        self,
        model_id: str,
    ) -> None:
        self.model_id = model_id

        self.pipeline = transformers.pipeline(
        "text-generation",
        model=self.model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",)

        self.terminators = [self.pipeline.tokenizer.eos_token_id, 
                            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    def prompt_list(self, system_prompt: str, user_prompt: List[str], temperature: float, top_p: float) -> str:
        """Using one system prompt together with different user prompts in the List.
        Args:
            system_prompt (str): system prompt
            user_prompt (List[str]): user prompts
            temperature (float): controls randomness
            top_p (float): controls randomness
        Returns:
            str: The LLM output text.
        """
        
        user_prompt = str(user_prompt)
        messages = self.__get_messages(system_prompt, user_prompt)

        return self.prompt(messages, temperature, top_p)
    
    def prompt_to_json(self, prompt: str, json_schema: dict, temperature: float) -> dict:
        """Generate a json given the schema and the prompt.
        Args:
            prompt (str): The prompt.
            json_schema (dict): JSON schema for output.
            temperature (float): controls randomness
        Returns:
            dict: LLM output strucutured as defined in JSON schema.
        """
        output = Jsonformer(self.pipeline.model, self.pipeline.tokenizer, json_schema, prompt, temperature=temperature)
        return output()

    def prompt(self, messages: List[dict], temperature: float, top_p: float) -> str:
        """Apply messages as defined in huggingface to prompt the LLM.
        Args:
            messages (List[dict]): The messages which include the prompts.
            temperature (float): Controls randomness.
            top_p (float): Controls randomness.
        Returns:
            str: the text output.
        """
        prompt = self.__get_prompt(messages)
        
        outputs = self.pipeline(prompt,
            max_new_tokens=256,
            eos_token_id=self.terminators,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        return outputs[0]["generated_text"][len(prompt):]

    def __get_prompt(self, messages: List[dict]) -> Union[List[int], Dict]:
        return self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def __get_messages(self, system_prompt: str, user_prompt: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]


class NER_Wrapper(object):
    """A wrapper for NER models using pretrained language models.
    Args:
            model_name_or_path (str): Local model dir of trained model or ID to load from huggingface.
    """
    def __init__(self, model_name_or_path: str, aggregation_strategy: str = "first") -> None:
        self.model_name_or_path = model_name_or_path
        self.aggregation_strategy = aggregation_strategy
        
        self.config =  AutoConfig.from_pretrained(self.model_name_or_path)
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name_or_path, config=self.config)
        self.pipeline = pipeline("ner", model=self.model, tokenizer=self.tokenizer, aggregation_strategy=self.aggregation_strategy)

    def __call__(self, texts: List[str]) -> List[List[Dict[str, Union[str, float, int]]]]:
        """Predict entities for list of strings.
        Args:
            texts (List[str]): input strings
        Returns:
            List[List[Dict[str, Union[str, float, int]]]]: entity maps
        """
        return self.pipeline(texts)
    
    def concat_entities(self, df: pd.DataFrame, text_attrs: List[str], extract_attrs: List[str], 
                        baseline_names: bool = True) -> pd.DataFrame:
        """Extract entites and concat them in columns to the input dataframe.
        Args:
            df (pd.DataFrame): input dataframe
            text_attrs (List[str]): col names which should be concatenated for entity extraction
            extract_attrs (List[str]): names of entities
            baseline_names (bool): whether to use baseline names of entities (eg. WoA and Artist). True by default
        Returns:
            pd.DataFrame: output dataframe
        """
        texts = get_concat(df, text_attrs)

        # predict using model
        ent_map = self(texts)

        # concat columns to df
        for raw_name in extract_attrs:
            
            key_name = BASELINE_NAMES[raw_name] if baseline_names else raw_name
            df[raw_name + '_ner'] = self.transform_entities(ent_map, key_name)
            
        return df

    @staticmethod
    def transform_entities(entities_raw: List[List[Dict[str, str]]], ent_name: str) -> List[List[str]]:
        """Transfrom raw output from pipeline to a list of lists with desired entities.
        Args:
            entities_raw (List[List[Dict[str, str]]]): output from pipeline
            ent_name (str): which entity to extract
        Returns:
            List[List[str]]: entities per item
        """
        all_extracted = []
        for ents in entities_raw:
            extracted = []
            for ent in ents:
                if ent["entity_group"] == ent_name:
                    extracted.append(ent["word"])
            all_extracted.append(extracted)
        return all_extracted
