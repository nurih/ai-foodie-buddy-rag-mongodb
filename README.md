
# Restaurant Reccomendations with AI and RAG architecture

This repo showcases using RAG architecture to produce a reasoned reccomendation in response to a user prompt.

## RAG Components

> :warning: You may need to create a Hugging Face account and authorize yourself to use and download models. See [this page https://huggingface.co/docs/hub/security-tokens](https://huggingface.co/docs/hub/security-tokens) for details.

### Embedding

For the sake of speed, the using [the sentance transformer multi-qa-MiniLM-L6-cos-v1](<https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1#multi-qa-minilm-l6-cos-v1>) performed well on a local machine using CPU only. The embedding vector length is only 384.

> :notebook: Changing to a different sentence transformer can yield different results during the retreival stage of the RAG. YMMV.

### LLM

For answer generation, this solution is using the `google/gemma-2b-it` LLM.

## Reccomendation Data

Restaurant meta data and corresponding reviews were loaded from source files provided [by this website](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).

The data is based on Google Local Data (2021), in a webpage by Tianyang Zhang, UCSD and Jiacheng Li, UCSD.

The data for a demo was prepared based on the Hawaii datasets for restaurant locations and reviews of those restaurants, further narrowed to places with 10 or more reviews and reviews that have a text length above 32 characters. This is under the theory that "Yay", and "Awesome", or "Garbage" represents a sentiment, but does little to describe the reasoning and therefore not helpful.

## What's in This Repo?

File | Purpose
--- | ---
`foodie_buddy_ai.py` | **Main program**. Demonstrates RAG architecture in practice
`embedder.py` | Utility that wraps an embedding model to generate embeddings for data preparation as well as for live query encoding.
`mongo_atlas.py` | Shared mongo code, instances of pre-connected collection handles
`reviews_extractor.py` | Utility that processes original raw data, making it read for use with Atlas vector search index
`reviews_uploader.py` | Utility that uploads data from files to MongoDB Atlas
`settings.py` | Shared settings / constants

## Python and local environment

> :x: In Progress - not fully implemented / tested.

If you run on an Intel laptop with Iris or other supported GPU, and environment with OpenVino can help speed things up a lot.

You will need to choose if to use OpenVino or the "vanilla" models, by instantiating the right wrapper:

Vanilla model | OpenVino optimized
--- | ---
`AutoModelForCausalLM.from_pretrained()` | `OVModelForCausalLM.from_pretrained('model-name', export=True)`

For that to work, you will need to consult and install the proper libraries

```shell
pip install --upgrade-strategy eager "optimum[openvino,nncf]"
pip install --upgrade-strategy eager "optimum[ipex]"
```
