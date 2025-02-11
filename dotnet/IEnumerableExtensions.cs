namespace SemanticKernelWithPostgres;

public static class IEnumerableExtensions
{
    public static IEnumerable<IEnumerable<T>> Batch<T>(this IEnumerable<T> source, int size)
    {
        T[] bucket = new T[size];
        var count = 0;

        foreach (var item in source)
        {
            bucket[count++] = item;

            if (count != size)
                continue;

            yield return bucket;

            bucket = new T[size];
            count = 0;
        }

        if (count > 0)
            yield return bucket.Take(count);
    }
}
