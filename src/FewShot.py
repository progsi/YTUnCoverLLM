import random
from typing import List
import os
from llama_index.core import PromptTemplate
from src.Utils import read_IOB_file, transform_to_dict
from src.Schema import EntityList, MusicEntity, Example
from src.Prompts import PROMPT_FEWSHOT_V3, PROMPT_FEWSHOT_V3_OUTPUT

def entity_dict_to_pydantic(entity_dict: dict) -> EntityList:
    """
    Args:
        entity_dict (dict): 
    Returns:
        EntityList: 
    """

    entity_list = EntityList(content=[])

    entity_dict["title"] = entity_dict.get("WoA")
    entity_dict["performer"] = entity_dict.get("Artist")
    for key in ["title", "performer"]:
        if entity_dict.get(key):
            for value in entity_dict[key]:
                entity_list.content.append(MusicEntity(utterance=value, label=key, cue=""))
    return entity_list

class FewShotSet:
    def __init__(self, test_path: str, output_instruction: bool = False) -> None:
        self.path = os.path.join(os.path.dirname(test_path), "train.bio")
        self.examples = self.__init_examples()
        self.output_instruction = output_instruction

    def __init_examples(self) -> List[Example]:
        """Init the examples list.
        Returns:
            List[Example]: _description_
        """

        texts_train, labels_train = read_IOB_file(self.path)

        examples = []
        for tokens, iob in zip(texts_train, labels_train):
            true_ents = transform_to_dict(tokens, iob)
            text = ' '.join(tokens)
            entity_list = entity_dict_to_pydantic(true_ents)
            examples.append(Example(text=text, output=entity_list))
        return examples

    def few_shot_examples_random(self, **kwargs) -> str:
        k = kwargs["k"]
        few_shot_examples = random.sample(self.examples, k)

        result_strs = []
        for example in few_shot_examples:
            result_str = f"""\
    Text: {example.text}
    Response: {example.output.model_dump()}"""
            result_strs.append(result_str)
        return "\n\n".join(result_strs)
    
    def get_prompt_template(self) -> PromptTemplate:
        """Get the few-shot prompt template based on the few-shot dataset.
        Returns:
            PromptTemplate: 
        """
        return PromptTemplate(
            PROMPT_FEWSHOT_V3 if not self.output_instruction else PROMPT_FEWSHOT_V3_OUTPUT,
            function_mappings={"few_shot_examples": self.few_shot_examples_random},
        )
