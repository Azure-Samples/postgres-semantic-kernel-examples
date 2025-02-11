using Microsoft.Extensions.Configuration;
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.AzureOpenAI;
using Microsoft.SemanticKernel.Connectors.Postgres;
using Microsoft.SemanticKernel.Data;
using Npgsql;
using CommandLine;

namespace SemanticKernelWithPostgres;

class Program
{
    private static IConfiguration? Configuration { get; set; }



    [Verb("load", HelpText = "Load ArXiv records into the database.")]
    public class LoadOptions
    {
        [Option('t', "total", Default = 50, HelpText = "Total number of results to load.")]
        public int Total { get; set; } = 50;

        [Option("topic", Default = "RAG", HelpText = "The topic to search ArXiv for.")]
        public string Topic { get; set; } = "RAG";
    }

    [Verb("query", HelpText = "Query the database.")]
    public class QueryOptions
    {
        [Value(0, MetaName = "query", Required = true, HelpText = "The query to search for.")]
        public string Query { get; set; } = "";
    }

    public static async Task Main(string[] args)
    {
        var builder = new ConfigurationBuilder()
            .SetBasePath(Directory.GetCurrentDirectory())
            .AddJsonFile("appsettings.json", optional: false, reloadOnChange: true)
            .AddEnvironmentVariables();

        Configuration = builder.Build();

        var parser = Parser.Default.ParseArguments<LoadOptions, QueryOptions>(args);

        await parser
            .WithParsedAsync<LoadOptions>(async options =>
            {
                await Load(options.Total, options.Topic);
            });
        await parser
            .WithParsedAsync<QueryOptions>(async options =>
            {
                await Query(options.Query);
            });

        parser
            .WithNotParsed(errors =>
            {
                Console.WriteLine("Invalid command line arguments.");
            });
    }

    private static NpgsqlDataSource GetDataSource()
    {
        var postgresConfig = Configuration!.GetSection("Postgres");
        var connectionString = postgresConfig["ConnectionString"] ?? throw new InvalidOperationException("Postgres connection string is missing.");
        var dataSourceBuilder = new NpgsqlDataSourceBuilder(connectionString);

        var useEntra = bool.Parse(postgresConfig["EntraAuthentication"] ?? "false");
        if (useEntra)
        {
            dataSourceBuilder.UseEntraAuthentication();
        }

        dataSourceBuilder.UseVector();
        return dataSourceBuilder.Build();
    }

    private static async Task Load(int totalResults = 1000, string topic = "RAG")
    {
        try
        {
            Console.WriteLine($"Loading ArXiv records for topic '{topic}'...");
            var records = await ArxivQuery.QueryArxivAsync(topic, totalResults: totalResults);
            Console.WriteLine($"Found {records.Count} results");

            var azureOpenAIConfig = Configuration!.GetSection("AzureOpenAI");
            if (azureOpenAIConfig is null)
            {
                throw new InvalidOperationException("AzureOpenAI configuration is missing.");
            }

            var textEmbeddingDeploymentName = azureOpenAIConfig["TextEmbeddingDeploymentName"] ?? throw new InvalidOperationException("TextEmbeddingDeploymentName is missing.");
            var endpoint = azureOpenAIConfig["Endpoint"] ?? throw new InvalidOperationException("Endpoint is missing.");
            var apiKey = azureOpenAIConfig["ApiKey"] ?? throw new InvalidOperationException("ApiKey is missing.");

            var textEmbeddingGenerationService = new AzureOpenAITextEmbeddingGenerationService(
                deploymentName: textEmbeddingDeploymentName,
                endpoint: endpoint,
                apiKey: apiKey);

            await using var dataSource = GetDataSource();
            var vectorStore = new PostgresVectorStore(dataSource);
            var recordCollection = vectorStore.GetCollection<string, ArxivRecord>("arxiv_records");
            await recordCollection.CreateCollectionIfNotExistsAsync().ConfigureAwait(false);

            // Group arxiv records into batches, generate embeddings for each batch, and upsert the records
            int i = 1;
            foreach (var batch in records.Batch(20))
            {
                Console.WriteLine($"Processing batch {i++} ({batch.Count()} records)...");
                var embeddings = await textEmbeddingGenerationService.GenerateEmbeddingsAsync(batch.Select(r => r.Abstract).ToList());
                Console.WriteLine("  ...embeddings generated");
                foreach (var zipped in batch.Zip(embeddings, (record, embedding) => (record, embedding)))
                {
                    zipped.record.Embedding = zipped.embedding;
                }

                await recordCollection.UpsertBatchAsync(batch).ToListAsync();
                Console.WriteLine("  ...batch upserted");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"An error occurred: {ex.Message}");
            throw;
        }
    }

    private static async Task Query(string query)
    {
        // Load the configuration
        var azureOpenAIConfig = Configuration!.GetSection("AzureOpenAI");
        if (azureOpenAIConfig is null)
        {
            throw new InvalidOperationException("AzureOpenAI configuration is missing.");
        }

        var textEmbeddingDeploymentName = azureOpenAIConfig["TextEmbeddingDeploymentName"] ?? throw new InvalidOperationException("TextEmbeddingDeploymentName is missing.");
        var endpoint = azureOpenAIConfig["Endpoint"] ?? throw new InvalidOperationException("Endpoint is missing.");
        var apiKey = azureOpenAIConfig["ApiKey"] ?? throw new InvalidOperationException("ApiKey is missing.");
        var chatCompletionDeploymentName = azureOpenAIConfig["ChatCompletionDeploymentName"] ?? throw new InvalidOperationException("ChatCompletionDeploymentName is missing.");

        // Create a text embedding service

        var textEmbeddingGenerationService = new AzureOpenAITextEmbeddingGenerationService(
                deploymentName: textEmbeddingDeploymentName,
                endpoint: endpoint,
                apiKey: apiKey);

        // Create a vector store and text search service

        await using var dataSource = GetDataSource();
        var vectorStore = new PostgresVectorStore(dataSource);
        var recordCollection = vectorStore.GetCollection<string, ArxivRecord>("arxiv_records");
        var textSearch = new VectorStoreTextSearch<ArxivRecord>(recordCollection, textEmbeddingGenerationService);

        // Create a kernel with OpenAI chat completion
        IKernelBuilder kernelBuilder = Kernel.CreateBuilder();
        kernelBuilder.AddAzureOpenAIChatCompletion(
            deploymentName: chatCompletionDeploymentName,
            endpoint: endpoint,
            apiKey: apiKey);

        Kernel kernel = kernelBuilder.Build();

        // Build a text search plugin with vector store search and add to the kernel
        var options = new KernelFunctionFromMethodOptions()
        {
            FunctionName = "ArXivSearch",
            Description = "Search for ArXiv abstracts for latest research in computer science topics.",
            Parameters =
            [
                new KernelParameterMetadata("query") { Description = "What to search for", IsRequired = true },
                new KernelParameterMetadata("count") { Description = "Number of results", IsRequired = true, DefaultValue = 3, ParameterType = typeof(int) },
                new KernelParameterMetadata("skip") { Description = "Number of results to skip", IsRequired = false, DefaultValue = 0 },
            ],
            ReturnParameter = new() { ParameterType = typeof(KernelSearchResults<string>) },
        };
        var searchPlugin = KernelPluginFactory.CreateFromFunctions("ArXiV", "Functions for working with research papers", [
            // textSearch.CreateGetSearchResults(options) // Requires https://github.com/microsoft/semantic-kernel/pull/10147
            textSearch.CreateGetSearchResultsCustom(options)
        ]);
        kernel.Plugins.Add(searchPlugin);

        AzureOpenAIPromptExecutionSettings settings = new() { FunctionChoiceBehavior = FunctionChoiceBehavior.Auto(), MaxTokens = 1000 };
        KernelArguments arguments = new(settings);
        Console.WriteLine(await kernel.InvokePromptAsync(query, arguments));
    }
}
