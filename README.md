# Reproducing-GPT2-small

<p align="center">
    <img src="assets/joke.png" alt="trainAnLLM" width="300" height="350">
    <img src="assets/understand.png" alt="understand" width="400" height="350">
</p>

Large Language Models (LLMs) can seem like dark magic. The idea of training a model like GPT2 might feel overwhelming, especially with all the hype and complexity surrounding the newest enterprise models. However, the best way to understand LLMs is to build one yourself.

This project is my own reimplementation of Andrej Karpathy’s work on [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master), breaking down the concepts into something approachable and practical for myself. By recreating GPT-2 step by step, I aim to gain a deeper understanding of how these models function under the hood.

## GPT2

We reimplemented the GPT2-small architecture according to the [repository](https://github.com/openai/gpt-2) and the [paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) released by OpenAI. The gpt2.py file contains the PyTorch implemantations of the GPT2 model which is mostly inspired by Karpathy's [build nanoGPT](https://github.com/karpathy/build-nanogpt) project. To verify the correctness of our architecture implementation, we load the pretrained weights from HuggingFace’s [GPT2LMHeadModel](https://huggingface.co/docs/transformers/v4.49.0/en/model_doc/gpt2#transformers.GPT2LMHeadModel) and ensure that the model behaves as expected.

For a visual interpretation of GPT2-small, explore the following [link](https://bbycroft.net/llm).

## FineWeb 

Unlike the original paper, which used [CommonCrawl](https://commoncrawl.org/) - dataset known for its large size but also significant noise—we prioritize data quality over quantity by using [FineWeb-edu](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1).

FineWeb-Edu is a high-quality dataset curated for pretraining large language models (LLMs), focusing on educational content. For nanoGPT-mini we use FineWeb-Edu-10B. The script for preparing the shards (fineweb.py) is exactly the same like in the [build nanoGPT](https://github.com/karpathy/build-nanogpt) project.
<p align="center">
    <img src="assets/fineweb.png" alt="understand" width="700" height="350">
</p>

## HellaSwag

HellaSwag provides a smooth evaluation and "early signals" (it slowly improves even for small models like ours). 

During the training, HellaSwag will be checked periodically. For each sample, we gradually shift tokens to compute the loss only over the candidate completion parts of the possible options. The option with the lowest loss will be selected.


## Pre-training

For pre-training, we used the same architecture size and hyper-parameters as presented in the [GPT3 paper](https://arxiv.org/pdf/2005.14165) (apparently GPT3 has more information regarding the experimental methodology than the original GPT2 paper). 

We ran the experiment on an RTX 4090 for about 48 hours on a cloud provider similar to Google Cloud Platform (GCP) and Amazon Web Services (AWS). 

## TODOs

* Post-train the current model with OpenAssistant dataset

## Some final notes

## Requirements 

