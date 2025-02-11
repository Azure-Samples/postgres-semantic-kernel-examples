// Copyright (c) Microsoft. All rights reserved.

namespace SemanticKernelWithPostgres;

/// <summary>
/// Constants for AzureDBForPostgres.
/// </summary>
public static class Constants
{
    /// <summary>
    /// The scope for the AzureDBForPostgres service, to be used with Entra.
    /// </summary>
    public const string AzureDBForPostgresScope = "https://ossrdbms-aad.database.windows.net/.default";
}