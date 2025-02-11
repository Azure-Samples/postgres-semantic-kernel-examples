using Microsoft.Extensions.VectorData;
using Microsoft.SemanticKernel.Data;

namespace SemanticKernelWithPostgres;

/// <summary>
/// Sample model class that represents a record entry.
/// </summary>
/// <remarks>
/// Note that each property is decorated with an attribute that specifies how the property should be treated by the vector store.
/// This allows us to create a collection in the vector store and upsert and retrieve instances of this class without any further configuration.
/// </remarks>
public sealed class ArxivRecord
{
    [VectorStoreRecordKey(StoragePropertyName = "id")]
    [TextSearchResultName]
    public string Id { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "title")]
    public string Title { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "abstract")]
    [TextSearchResultValue]
    public string Abstract { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "published")]
    public DateTime Published { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "authors")]
    public List<string> Authors { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "categories")]
    public List<string> Categories { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "link")]
    [TextSearchResultLink]
    public string Link { get; init; }

    [VectorStoreRecordData(StoragePropertyName = "pdf_link")]
    public string PdfLink { get; init; }

    [VectorStoreRecordVector(Dimensions: 1536, StoragePropertyName = "embedding")]
    public ReadOnlyMemory<float>? Embedding { get; set; }
}
