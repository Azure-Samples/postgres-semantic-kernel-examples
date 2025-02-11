## Postgres & Python Semantic Kernel Examples

### Setup

To set up the project, you need to install the required dependencies. You can do this by running the following command in the `python` directory:

```sh
pip install -e .
```

### Configuration

Before running the examples, you need to configure your database and Azure OpenAI settings. Use the `.env.example` file in the `python` directory as a template to create a `.env` file with the proper settings for your database. Update the connection details for your PostgreSQL database and Azure OpenAI instance in the `.env` file.

### Using Azure DB for PostgreSQL

This project includes connection logic to use Azure DB for PostgreSQL with Entra authentication. If the configured connection string does not contain a user or password, Entra Authentication is used with DefaultAzureCredentials. You'll need to be logged into the Azure CLI to correctly authenticate with your Entra credentials.

### Vector Store Retrieval Augmented Generation (RAG)

The `vector_store_rag` package contains an example of using the Postgres vector store as a memory connector for a RAG example. This example demonstrates how to search the arXiv API for specific papers based on a topic and category, and then use those papers to perform various tasks such as searching and chatting with the model. 

#### Running the Example

To run the example, navigate to the `python` directory and execute the following commands:

##### Load Data

```sh
pg-sk-examples rag load --num-papers 100 --topic RAG --category cs.AI --env-file-path .env
```

This will load the data from arXiv papers based on the specified topic and category. The `category` parameter should be one of the arXiv categories, which you can find [here](https://arxiv.org/category_taxonomy). The abstracts of the papers are then embedded using Azure OpenAI's embedding service, and these embeddings are stored in a PostgreSQL vector store for efficient searching.

##### Search Data

```sh
pg-sk-examples rag search "Your query here" --count 5
```

This will search the loaded arXiv papers based on the provided query by comparing the query embedding against the stored embeddings in the PostgreSQL vector store.

##### Chat with the Model

```sh
pg-sk-examples rag chat
```

This will start a chat session with the model using the loaded arXiv papers.

### Additional Information

For more details on the project and its usage, refer to the comments and documentation within the codebase.
