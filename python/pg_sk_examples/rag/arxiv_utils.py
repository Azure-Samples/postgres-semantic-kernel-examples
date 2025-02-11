import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import numpy as np
import requests
from pydantic import BaseModel
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIEmbeddingPromptExecutionSettings,
)
from semantic_kernel.data.const import DistanceFunction, IndexKind
from semantic_kernel.data.record_definition.vector_store_model_decorator import vectorstoremodel
from semantic_kernel.data.record_definition.vector_store_record_fields import (
    VectorStoreRecordDataField,
    VectorStoreRecordKeyField,
    VectorStoreRecordVectorField,
)


@vectorstoremodel
class ArxivPaper(BaseModel):
    id: Annotated[str, VectorStoreRecordKeyField()]
    title: Annotated[str, VectorStoreRecordDataField()]
    abstract: Annotated[str, VectorStoreRecordDataField(has_embedding=True, embedding_property_name="embedding")]
    published: Annotated[datetime, VectorStoreRecordDataField()]
    authors: Annotated[list[str], VectorStoreRecordDataField()]
    categories: Annotated[list[str], VectorStoreRecordDataField()]
    link: Annotated[str | None, VectorStoreRecordDataField()]
    pdf_link: Annotated[str | None, VectorStoreRecordDataField()]

    embedding: Annotated[
        list[float] | None,
        VectorStoreRecordVectorField(
            embedding_settings={"embedding": OpenAIEmbeddingPromptExecutionSettings(dimensions=1536)},
            index_kind=IndexKind.HNSW,
            dimensions=1536,
            distance_function=DistanceFunction.COSINE_DISTANCE,
            property_type="float",
        ),
    ] = None

    @classmethod
    def from_arxiv_info(cls, arxiv_info: dict[str, Any]) -> "ArxivPaper":
        return cls(
            id=arxiv_info["id"],
            title=arxiv_info["title"].replace("\n  ", " "),
            abstract=arxiv_info["abstract"].replace("\n  ", " "),
            published=arxiv_info["published"],
            authors=arxiv_info["authors"],
            categories=arxiv_info["categories"],
            link=arxiv_info["link"],
            pdf_link=arxiv_info["pdf_link"],
        )


def query_arxiv(
    search_query: str, category: str = "cs.AI", page_size: int = 100, total_results: int = 100
) -> list[dict[str, Any]]:
    """Query the ArXiv API and return a list of dictionaries with relevant metadata for each paper.

    Args:
        search_query: The search term or topic to query for.
        category: The category to restrict the search to (default is "cs.AI").
        See https://arxiv.org/category_taxonomy for a list of categories.
        page_size: Number of results per page (default is 100).
        total_results: Total number of results to retrieve (default is 100).
    """
    all_results = []
    current_start = 0
    api_max_results = 2000  # The maximum results Arxiv allows per query
    page_size = min(page_size, api_max_results)  # Ensure page_size per query adheres to Arxiv limits
    remaining_results = total_results

    while remaining_results > 0:
        # Adjust page size for the final request if remaining results are less than page_size
        current_max_results = min(page_size, remaining_results)

        # Construct the query URL
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query=all:%22{search_query.replace(' ', '+')}%22"
            f"+AND+cat:{category}&start={current_start}&max_results={current_max_results}&sortBy=lastUpdatedDate&sortOrder=descending"
        )

        # Fetch the data
        response = requests.get(url)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}

        # Extract relevant metadata
        results = [
            {
                "id": entry.find("atom:id", ns).text.split("/")[-1],
                "title": entry.find("atom:title", ns).text,
                "abstract": entry.find("atom:summary", ns).text,
                "published": entry.find("atom:published", ns).text,
                "link": entry.find("atom:id", ns).text,
                "authors": [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)],
                "categories": [category.get("term") for category in entry.findall("atom:category", ns)],
                "pdf_link": next(
                    (
                        link_tag.get("href")
                        for link_tag in entry.findall("atom:link", ns)
                        if link_tag.get("title") == "pdf"
                    ),
                    None,
                ),
            }
            for entry in root.findall("atom:entry", ns)
        ]

        # Add to all results
        all_results.extend(results)

        # Update pagination parameters
        current_start += current_max_results
        remaining_results -= current_max_results

        # Delay 3 seconds before making the next request, as per Arxiv API guidelines
        if remaining_results > 0:
            import time

            time.sleep(3)

    return all_results
