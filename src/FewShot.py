import random
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
        self.masked_texts = [self.get_masked_text(example) for example in self.examples]
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
    
    def __get_similar_examples(self, text: str, k: int) -> List[Example]:
        """Get similar examples based on tfidf
        Args:
            text (str): 
            k (int): 
        Returns:
            List[Example]: 
        """
        # Create a TF-IDF Vectorizer
        vectorizer = TfidfVectorizer()
        all_texts = [text] + self.masked_texts
        
        # Transform texts to TF-IDF vectors
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute cosine similarities between the query and the examples
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        
        # Get indices of the top k most similar examples
        similar_indices = cosine_similarities[0].argsort()[-k:][::-1]
        
        # Return the top k most similar examples
        return [self.examples[i] for i in similar_indices]

    def few_shot_examples_similar(self, **kwargs) -> str:
        """
        Find the k most similar examples based on masked text similarity.
        Args:
            examples (Example): A variable number of Example instances.
            k (int): Number of similar examples to return.
            text (str): The text to compare against.
        Returns:
            List[Example]: The k most similar Example instances.
        """
        k, text = kwargs["k"], kwargs["text"]
        # Get the masked text of the input query
        few_shot_examples = self.__get_similar_examples(text=text, k=k)

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

    @staticmethod
    def get_masked_text(example: Example) -> str:
        """Get text with masked entities.
        Args:
            example (Example): An instance of Example class with text and entities.
        Returns:
            str: Text with entities replaced by a placeholder including their label.
        """
        # Extract text and entities from the example
        text = example.text
        entities = example.output.content
        
        # Sort entities by their start position in the text (from end to start to avoid index shifting)
        entities = sorted(entities, key=lambda e: text.find(e.utterance), reverse=True)
        
        # Mask the entities in the text
        masked_text = text
        for entity in entities:
            mask = f"[{entity.label.upper()}]"
            start_index = text.find(entity.utterance)
            end_index = start_index + len(entity.utterance)
            masked_text = masked_text[:start_index] + mask + masked_text[end_index:]
        
        return masked_text