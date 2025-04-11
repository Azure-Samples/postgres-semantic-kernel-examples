import textwrap

from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.azure_text_embedding import AzureTextEmbedding
from semantic_kernel.connectors.memory.postgres import PostgresCollection, PostgresSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.data.vector_search import add_vector_to_records
from semantic_kernel.data.text_search.vector_store_text_search import VectorStoreTextSearch
from semantic_kernel.data.vector_search.vector_search_options import VectorSearchOptions
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions.kernel_parameter_metadata import KernelParameterMetadata
from semantic_kernel.kernel import Kernel

from ..entra_connection import AsyncEntraConnection
from .arxiv_utils import ArxivPaper, query_arxiv


async def load_arxiv_papers(
    total_papers: int = 100, topic: str = "RAG", category: str = "cs.AI", env_file_path: str = ".env"
):
    arxiv_papers: list[ArxivPaper] = [
        ArxivPaper.from_arxiv_info(paper) for paper in query_arxiv(topic, category=category, total_results=total_papers)
    ]

    print(f"Found {len(arxiv_papers)} papers on '{topic}'")

    # Create a Kernel
    kernel = Kernel()

    # Add the TextEmbedding service, which will be used to generate embeddings of the abstract for each paper.
    text_embedding = AzureTextEmbedding(service_id="embedding", env_file_path=env_file_path)
    kernel.add_service(text_embedding)

    # Create a connection pool to use with the PostgresCollection
    pg_settings = PostgresSettings.create(env_file_path=env_file_path)
    print(f"Creating connection pool to {pg_settings.get_connection_args()['host']}...")
    connection_pool = await pg_settings.create_connection_pool(connection_class=AsyncEntraConnection)
    async with connection_pool:
        collection = PostgresCollection[str, ArxivPaper](
            collection_name="arxiv_records",
            data_model_type=ArxivPaper,
            env_file_path=env_file_path,
            connection_pool=connection_pool,
        )

        # Create the collection if it doesn't exist
        await collection.create_collection_if_not_exists()

        # Process arxiv papers in batches of 20
        for i in range(0, len(arxiv_papers), 20):
            # Add embeddings to the abstracts of the papers
            records = await add_vector_to_records(kernel, 
                arxiv_papers[i : i + 20], data_model_type=ArxivPaper
            )
            # Upsert the records into the collection
            await collection.upsert_batch(records)
            print(f"...Loaded {i + 20} papers into the collection")


async def search_arxiv_papers(query: str, count: int = 3, env_file_path: str = ".env"):
    # Create a Kernel
    kernel = Kernel()

    # Add the TextEmbedding service, which will be used to generate embeddings of the abstract for each paper.
    text_embedding = AzureTextEmbedding(service_id="embedding", env_file_path=env_file_path)
    kernel.add_service(text_embedding)

    # Create a connection pool to use with the PostgresCollection
    settings = PostgresSettings.create(env_file_path=env_file_path)
    connection_pool = await settings.create_connection_pool(connection_class=AsyncEntraConnection)
    async with connection_pool:
        collection = PostgresCollection[str, ArxivPaper](
            collection_name="arxiv_records",
            data_model_type=ArxivPaper,
            env_file_path=env_file_path,
            connection_pool=connection_pool,
        )

        text_search = VectorStoreTextSearch[ArxivPaper].from_vectorized_search(
            collection, embedding_service=text_embedding
        )

        search_results = await text_search.get_search_results(
            query, options=VectorSearchOptions(top=count, include_total_count=True)
        )
        print()
        print(f"Found {search_results.total_count} papers.")
        print()
        async for result in search_results.results:
            axriv_record = result.record
            score = result.score

            print(f"Title: {axriv_record.title}")
            print(f"Link: {axriv_record.link}")
            print(f"Score: {score}")
            print()


async def chat_with_arxiv_papers(env_file_path: str = ".env"):
    # Create a Kernel
    kernel = Kernel()

    # Add the TextEmbedding service, which will be used to generate embeddings of the abstract for each paper.
    text_embedding = AzureTextEmbedding(service_id="embedding", env_file_path=env_file_path)
    kernel.add_service(text_embedding)

    # Add the AzureChatCompletion service, which will be used to generate completions for the chat.
    chat_completion = AzureChatCompletion(service_id="chat", deployment_name="gpt-4o")
    kernel.add_service(chat_completion)

    # Create a connection pool to use with the PostgresCollection
    settings = PostgresSettings.create(env_file_path=env_file_path)
    connection_pool = await settings.create_connection_pool(connection_class=AsyncEntraConnection)
    async with connection_pool:
        collection = PostgresCollection[str, ArxivPaper](
            collection_name="arxiv_records",
            data_model_type=ArxivPaper,
            env_file_path=env_file_path,
            connection_pool=connection_pool,
        )

        text_search = VectorStoreTextSearch[ArxivPaper].from_vectorized_search(
            collection, embedding_service=text_embedding
        )

        # Add the Azure AI Search plugin to the kernel
        kernel.add_functions(
            plugin_name="arxiv_plugin",
            functions=[
                text_search.create_search(
                    # The default parameters match the parameters of the VectorSearchOptions class.
                    description="Searches for ArXiv papers that are related to the query.",
                    parameters=[
                        KernelParameterMetadata(
                            name="query",
                            description="What to search for.",
                            type="str",
                            is_required=True,
                            type_object=str,
                        ),
                        KernelParameterMetadata(
                            name="top",
                            description="Number of results to return.",
                            type="int",
                            default_value=2,
                            type_object=int,
                        ),
                    ],
                ),
            ],
        )

        chat_function = kernel.add_function(
            prompt="{{$chat_history}}{{$user_input}}",
            plugin_name="ChatBot",
            function_name="Chat",
        )

        # we set the function choice to Auto, so that the LLM can choose the correct function to call.
        # and we exclude the ChatBot plugin, so that it does not call itself.
        # this means that it has access to 2 functions, that were defined above.
        execution_settings = AzureChatPromptExecutionSettings(
            function_choice_behavior=FunctionChoiceBehavior.Auto(filters={"excluded_plugins": ["ChatBot"]}),
            service_id="chat",
            max_tokens=7000,
            temperature=0.7,
            top_p=0.8,
        )

        history = ChatHistory()
        system_message = """
        You are a chat bot. Your name is Archie and
        you have one goal: help people find answers
        to technical questions by relying on the latest
        research papers published on ArXiv.
        You communicate effectively in the style of a helpful librarian. 
        You always make sure to include the
        ArXiV paper references in your responses.
        If you cannot find the answer in the papers,
        you will let the user know, but also provide the papers
        you did find to be most relevant. If the abstract of the 
        paper does not specifically reference the user's inquiry,
        but you believe it might be relevant, you can still include it
        BUT you must make sure to mention that the paper might not directly
        address the user's inquiry. Make certain that the papers you link are
        from a specific search result.
        """
        history.add_system_message(system_message)
        history.add_user_message("Hi there, who are you?")
        history.add_assistant_message(
            "I am Archie, the ArXiV chat bot. "
            "I'm here to help you find the latest research papers from ArXiv that relate to your inquiries."
        )

        arguments = KernelArguments(settings=execution_settings)

        def wrap_text(text, width=90):
            paragraphs = text.split("\n\n")  # Split the text into paragraphs
            wrapped_paragraphs = [
                "\n".join(
                    textwrap.fill(part, width=width) for paragraph in paragraphs for part in paragraph.split("\n")
                )
            ]  # Wrap each paragraph, split by newlines
            return "\n\n".join(wrapped_paragraphs)  # Join the wrapped paragraphs back together

        chatting = True
        while chatting:
            try:
                user_input = input("User:> ")
            except KeyboardInterrupt:
                print("\n\nExiting chat...")
                return
            except EOFError:
                print("\n\nExiting chat...")
                return

            if user_input == "exit":
                print("\n\nExiting chat...")
                return
            arguments["user_input"] = user_input
            arguments["chat_history"] = history
            result = await kernel.invoke(chat_function, arguments=arguments)
            print()
            print(f"Archie:>\n\n{wrap_text(str(result))}")
            print()
            history.add_user_message(user_input)
            history.add_assistant_message(str(result))
            chatting = True
