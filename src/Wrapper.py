import transformers
import torch
from typing import List
from jsonformer import Jsonformer


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

    def prompt_list(self, system_prompt: str, user_prompt: List[str], temperature: float, top_p: float):
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
    
    def prompt_to_json(self, prompt: str, json_schema: dict, temperature: float):
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

    def prompt(self, messages: List[dict], temperature: float, top_p: float):
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

    def __get_prompt(self, messages: List[dict]):
        return self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    def __get_messages(self, system_prompt: str, user_prompt: str):
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}]

