# MM-RAG Demo

This repository contains a demonstration of the Multi-Modal Retrieval-Augmented Generation (MM-RAG) model, designed for multi-modal tasks. MM-RAG extends the original RAG model to handle various modalities such as text, images, and possibly more, by retrieving relevant content from a knowledge base and generating outputs conditioned on these retrieved elements.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Example](#example)
- [Model Architecture](#model-architecture)
- [License](#license)

## Overview

MM-RAG is a powerful model that combines the strengths of retrieval-based and generative models. It is particularly suited for tasks that involve multiple data types, such as text and images. This demo illustrates the model's ability to retrieve relevant information from a large corpus and generate responses based on this multi-modal data.

## Features

- **Multi-Modal Input:** Handles and processes both textual and visual data.
- **Retrieval-Augmented Generation:** Retrieves relevant documents/images and conditions the generation process on this retrieved information.
- **Flexibility:** Can be adapted to various applications, including multi-modal question answering, image captioning, and more.

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch
- Transformers
- Other dependencies listed in `requirements.txt`

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/aaviix/MM-RAG-Demo.git
   cd MM-RAG-Demo
   ```
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Set up the environment:

   - Download the required pre-trained models and datasets as specified in the setup instructions.
4. Run the application:

   ```bash
   streamlit run app.py
   ```

## Example

Here is an example of how to use the MM-RAG Demo:

1. **Text Query:**

   - Input: "What is the capital of France?"
   - Output: "The capital of France is Paris."
2. **Image Query:**

   - Input: An image of the Eiffel Tower.
   - Output: "This is the Eiffel Tower located in Paris, France."

## Model Architecture

The MM-RAG model is built on top of the Retrieval-Augmented Generation (RAG) framework. It consists of two main components:

1. **Retriever:** Fetches relevant documents or images based on the input query.
2. **Generator:** Produces the final output conditioned on the retrieved content.

The model supports various retrieval mechanisms and can be customized for different types of multi-modal tasks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
