## PostgreSQL & Semantic Kernel: Examples

### Introduction

This repository contains examples and sample code for using SemanticKernel with PostgreSQL in general and Azure DB for PostgreSQL in particular.

### Docker setup

Included is a `docker-compose.yml` file that sets up a PostgreSQL with the `pgvector` extension installed. This can be used to test the examples in this repository
in a local setup. To run, make sure you have Docker installed and run:

```bash
docker-compose up
```

You can now connect to the `postgres` database on `localhost:5432` with the username `postgres` and password `example`.

### .NET Examples

The `dotnet` folder contains examples of using Postgres with the .NET SemanticKernel library. See [dotnet/README.md](dotnet/README.md) for more details.

- **Arxiv Query**: Demonstrates how to query the arXiv API for papers and process the results.
- **Text Search Extensions**: Provides extensions for performing text searches on the data.

### Python Examples

The `python` folder contains examples of using Postgres with the Python SemanticKernel library. See [python/README.md](python/README.md) for more details.

- **Vector Store Retrieval Augmented Generation (RAG)**: Demonstrates how to search the arXiv API for specific papers based on a topic and category, embed the abstracts using Azure OpenAI's embedding service, and store these embeddings in a PostgreSQL vector store for efficient searching. It includes commands for loading data, searching data, and chatting with the model.
