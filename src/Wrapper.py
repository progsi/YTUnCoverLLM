import os
import re
from typing import List, Dict, Union
from jsonformer import Jsonformer
from transformers import pipeline, AutoConfig, AutoTokenizer, AutoModelForTokenClassification, AutoModelForCausalLM
import pandas as pd
from src.Utils import get_concat_col_name, get_left_right_concat, get_concat
from preprocessing.Utils import BASELINE_NAMES


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

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Most LLMs don't have a pad token by default
    
    def prompt_to_json(self, prompt: str, json_schema: dict, temperature: float) -> dict:
        """Generate a json given the schema and the prompt.
        Args:
            prompt (str): The prompt.
            json_schema (dict): JSON schema for output.
            temperature (float): controls randomness
        Returns:
            dict: LLM output strucutured as defined in JSON schema.
        """
        output = Jsonformer(self.model, self.tokenizer, json_schema, prompt, temperature=temperature)
        return output()

    def prompt(self, system_prompt: str, user_prompts: List[str], temperature: float = 0.0, top_p: float = 1.0, top_k: int = 50) -> str:
        """Apply messages as defined in huggingface to prompt the LLM.
        Args:
            system_prompt (str): system prompt
            user_prompts (List[str]): user prompts
            temperature (float): Controls randomness.
            top_p (float):  If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        Returns:
            str: the text output.
        """
        prompts = [system_prompt + user_prompt for user_prompt in user_prompts]

        model_inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
        generated_ids = self.model.generate(**model_inputs, temperature=temperature, top_p=top_p, top_k=top_k)
        #outputs = self.tokenizer.batch_decode(generated_ids[:, model_inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return outputs


class NER_Wrapper(object):
    """A wrapper for NER models using pretrained language models.
    Args:
            model_name (str): Local model dir of trained model or ID to load from huggingface.
    """
    def __init__(self, model_name: str, aggregation_strategy: str = "first") -> None:
        self.model_name = model_name
        self.aggregation_strategy = aggregation_strategy
        self.OUTPUT_SUBDIR = "shs100k2" # adjust if necessary! subdir in the output dir in music-ner-eacl2023
        
        self.config =  AutoConfig.from_pretrained(self.model_name)
        self.tokenizer =  AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_name, config=self.config)
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
    
    def get_path(self, checkpoint: int = None) -> str:
        """Get path of pretrained model.
        Args:
            checkpoint (int, optional): Checkpoint to load. Defaults to None --> load last.
        Returns:
            str: the path
        """
        base_path = self.get_base_path()
        if checkpoint == None:
            last_checkpoint = f"checkpoint-{self.get_last_checkpoint(base_path)}"
            return os.path.join(base_path, last_checkpoint)
        else:
            return os.path.join(base_path, checkpoint)
    
    def get_base_path(self) -> str:
        """Get base path of model.
        Returns:
            str: base path
        """
        return os.path.join("..", "baseline", "music-ner-eacl2023", "output", self.OUTPUT_SUBDIR, self.model_name)

    def get_last_checkpoint(self) -> str:
        """Get last checkpoint by directory names in given dir.
        Returns:
            str: last checkpoint number
        """
        checkpoint_pattern = re.compile(r'^checkpoint-(\d+)$')
        last_checkpoint = None
        # List all items in the directory
        for item in os.listdir(self.get_base_path()):
            # Check if the item matches the checkpoint pattern
            match = checkpoint_pattern.match(item)
            if match:
                # Extract the checkpoint number
                checkpoint_number = int(match.group(1))
                # Update last_checkpoint if this one is higher
                if last_checkpoint is None or checkpoint_number > last_checkpoint:
                    last_checkpoint = checkpoint_number
        return last_checkpoint
