using System.Xml.Linq;

namespace SemanticKernelWithPostgres;

class ArxivQuery
{
    private static readonly HttpClient httpClient = new HttpClient();

    public static async Task<List<ArxivRecord>> QueryArxivAsync(string searchQuery, string category = "cs.AI", int pageSize = 100, int totalResults = 100)
    {
        var allResults = new List<ArxivRecord>();
        int currentStart = 0;
        const int apiMaxResults = 2000; // The maximum results Arxiv allows per query
        pageSize = Math.Min(pageSize, apiMaxResults); // Ensure pageSize per query adheres to Arxiv limits
        int remainingResults = totalResults;

        while (remainingResults > 0)
        {
            // Adjust page size for the final request if remaining results are less than pageSize
            int currentMaxResults = Math.Min(pageSize, remainingResults);

            // Construct the query URL
            string url = $"http://export.arxiv.org/api/query?search_query=all:%22{Uri.EscapeDataString(searchQuery)}%22+AND+cat:{category}" +
                         $"&start={currentStart}&max_results={currentMaxResults}&sortBy=lastUpdatedDate&sortOrder=descending";

            // Fetch the data
            HttpResponseMessage response = await httpClient.GetAsync(url);
            response.EnsureSuccessStatusCode();
            string responseContent = await response.Content.ReadAsStringAsync();

            // Parse the XML response
            XDocument xmlDoc = XDocument.Parse(responseContent);
            XNamespace atom = "http://www.w3.org/2005/Atom";

            // Extract relevant metadata
            var results = xmlDoc.Descendants(atom + "entry")
                .Select(entry => new ArxivRecord
                {
                    Id = entry.Element(atom + "id")?.Value.Split('/').Last(),
                    Title = entry.Element(atom + "title")?.Value,
                    Abstract = entry.Element(atom + "summary")?.Value,
                    Published = DateTime.Parse(entry.Element(atom + "published")?.Value),
                    Link = entry.Element(atom + "id")?.Value,
                    Authors = entry.Elements(atom + "author").Select(author => author.Element(atom + "name")?.Value).ToList(),
                    Categories = entry.Elements(atom + "category").Select(cat => cat.Attribute("term")?.Value).ToList(),
                    PdfLink = entry.Elements(atom + "link")
                                   .Where(link => link.Attribute("title")?.Value == "pdf")
                                   .Select(link => link.Attribute("href")?.Value)
                                   .FirstOrDefault()
                })
                .ToList();

            // Add to all results
            allResults.AddRange(results);

            // Update pagination parameters
            currentStart += currentMaxResults;
            remainingResults -= currentMaxResults;

            // Delay 3 seconds before making the next request, as per Arxiv API guidelines
            if (remainingResults > 0) await Task.Delay(3000);
        }

        return allResults;
    }
}
