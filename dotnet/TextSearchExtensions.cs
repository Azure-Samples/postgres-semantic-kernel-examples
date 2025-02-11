using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Data;

namespace SemanticKernelWithPostgres;

public static class TextSearchExtensions
{
    /// <summary>
    /// Custom CreateGetSearchResults method that can be removed after this fix: https://github.com/microsoft/semantic-kernel/pull/10147
    public static KernelFunction CreateGetSearchResultsCustom(this ITextSearch textSearch, KernelFunctionFromMethodOptions options, TextSearchOptions? searchOptions = null)
    {
        int GetArgumentValue(KernelArguments arguments, IReadOnlyList<KernelParameterMetadata> parameters, string name, int defaultValue)
        {
            if (arguments.TryGetValue(name, out var value))
            {
                if (value is int argument)
                {
                    return argument;
                }
                else if (value is string argumentString && int.TryParse(argumentString, out var parsedArgument))
                {
                    return parsedArgument;
                }
            }

            value = parameters.FirstOrDefault(parameter => parameter.Name == name)?.DefaultValue;
            if (value is int metadataDefault)
            {
                return metadataDefault;
            }

            return defaultValue;
        }

        async Task<IEnumerable<object>> GetSearchResultAsync(Kernel kernel, KernelFunction function, KernelArguments arguments, CancellationToken cancellationToken)
        {
            arguments.TryGetValue("query", out var query);
            if (string.IsNullOrEmpty(query?.ToString()))
            {
                return [];
            }

            var parameters = function.Metadata.Parameters;

            searchOptions ??= new()
            {
                Top = GetArgumentValue(arguments, parameters, "count", 2),
                Skip = GetArgumentValue(arguments, parameters, "skip", 0)
            };

            var result = await textSearch.GetSearchResultsAsync(query?.ToString()!, searchOptions, cancellationToken).ConfigureAwait(false);
            return await result.Results.ToListAsync(cancellationToken).ConfigureAwait(false);
        }

        return KernelFunctionFactory.CreateFromMethod(
                GetSearchResultAsync,
                options);
    }
}
