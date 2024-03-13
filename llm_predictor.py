from enum import StrEnum
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import vertexai
from vertexai import language_models
import os

from settings import LLM_PROMPT_CACHE


GEMMA = "google/gemma-2b-it"


class VertexModel(StrEnum):
    TEXT_BISON = "text-bison@002"
    TEXT_UNICORN = "text-unicorn@001"


class LlmPredictor:

    def use_api_vertex_ai(
        self,
        model_name: str = VertexModel.TEXT_BISON,
        gcp_project: str = None,
        gcp_location: str = None,
    ):

        print("Using Vertex AI for prediction", vertexai.aiplatform_version.__version__)

        # gather environment variables as necessary

        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ[
            "GCP_SERVICE_ACCOUNT_KEY_PATH"
        ]

        gcp_project or os.environ["GCP_PROJECT"]

        gcp_location or os.environ["GCP_LOCATION"]

        # Initialize vertexai
        vertexai.init(project=gcp_project, location=gcp_location)

        self._generate = lambda prompt: _predict_vertex_ai(prompt, model_name)

        return self

    def use_local_gemma(self):

        print("Using Gemma local model")

        print("Loading tokenizer")

        tokenizer = AutoTokenizer.from_pretrained(GEMMA)

        print("Loading Language Model")

        model = AutoModelForCausalLM.from_pretrained(GEMMA, device_map="auto")

        self._generate = lambda prompt: _predict_gemma(prompt, tokenizer, model)
        return self

    def predict(self, prompt):
        return self._generate(prompt)


def _predict_gemma(llm_prompt: str, tokenizer, model) -> str:

    # tokenize prompt
    input_ids = tokenizer(llm_prompt, return_tensors="pt")

    # generate response as tensor
    tensore_response: torch.Tensor = model.generate(**input_ids, max_length=512)

    # decode tensor into text
    decoded_response: str = tokenizer.decode(tensore_response[0])

    # Trim away original prompt from generated text
    result = decoded_response.replace(llm_prompt, "")

    return result


def _predict_vertex_ai(prompt: str, model_name: str) -> str:
    parameters = {
        "temperature": 0.17,
        "max_output_tokens": 160,
        "top_p": 0.86,
        "top_k": 40,
    }

    model = language_models.TextGenerationModel.from_pretrained(model_name)

    response = model.predict(
        prompt,
        **parameters,
    )
    return response.text


if __name__ == "__main__":

    llm = LlmPredictor().use_api_vertex_ai(model_name=VertexModel.TEXT_BISON)
    llm = LlmPredictor().use_api_vertex_ai(model_name=VertexModel.TEXT_UNICORN)
    # llm = LlmOneShot().use_local_gemma()

    prompt = open(LLM_PROMPT_CACHE, "r").read()

    print("#" * 80)
    print(prompt)
    print("#" * 80)
    print(llm.predict(prompt))
