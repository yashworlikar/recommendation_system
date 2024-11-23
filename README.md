---
title: Fashion Recommendation
emoji: 👚
colorFrom: yellow
colorTo: purple
sdk: gradio
sdk_version: 5.5.0
app_file: app.py
pinned: false
license: mit
short_description: Fashion Recommendation
---

A basic recommendation system using text-vector embeddings based on [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)

The embeddings are generate using the [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) model.
This is a text only embedding model, so the product images are not considered when generating search results.

