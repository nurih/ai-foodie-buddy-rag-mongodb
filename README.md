
# Restaurant Reccomendations with AI and RAG architecture

This repo showcases using RAG architecture to produce a reasoned reccomendation in response to a user prompt.

## RAG Components

### Embedding

For the sake of speed, the using [the sentance transformer multi-qa-MiniLM-L6-cos-v1](<https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1#multi-qa-minilm-l6-cos-v1>) performed well on a local machine using CPU only. The embedding vector length is only 384.

> Changing to a different sentence transformer can yield different results during the retreival stage of the RAG. YMMV.

### LLM

For answer generation, this solution is using the `google/gemma-2b-it` LLM.

## Reccomendation Data

Restaurant meta data and corresponding reviews were loaded from source files provided [by this website](https://datarepo.eng.ucsd.edu/mcauley_group/gdrive/googlelocal/).

The data is based on Google Local Data (2021), in a webpage by Tianyang Zhang, UCSD and Jiacheng Li, UCSD.

The data for a demo was prepared based on the Hawaii datasets for restaurant locations and reviews of those restaurants, further narrowed to places with 10 or more reviews and reviews that have a text length above 32 characters. This is under the theory that "Yay", and "Awesome", or "Garbage" represents a sentiment, but does little to describe the reasoning and therefore not helpful.
