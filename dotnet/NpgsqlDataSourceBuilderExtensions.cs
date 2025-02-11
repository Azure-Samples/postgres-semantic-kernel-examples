// Copyright (c) Microsoft. All rights reserved.

using System.Text;
using System.Text.Json;
using Azure.Core;
using Azure.Identity;
using Npgsql;

namespace SemanticKernelWithPostgres;

public static class NpgsqlDataSourceBuilderExtensions
{
    private static readonly TokenRequestContext s_azureDBForPostgresTokenRequestContext = new([Constants.AzureDBForPostgresScope]);

    public static NpgsqlDataSourceBuilder UseEntraAuthentication(this NpgsqlDataSourceBuilder dataSourceBuilder, TokenCredential? credential = default)
    {
        credential ??= new DefaultAzureCredential();

        if (dataSourceBuilder.ConnectionStringBuilder.Username == null)
        {
            var token = credential.GetToken(s_azureDBForPostgresTokenRequestContext, default);
            SetUsernameFromToken(dataSourceBuilder, token.Token);
        }

        SetPasswordProvider(dataSourceBuilder, credential, s_azureDBForPostgresTokenRequestContext);

        return dataSourceBuilder;
    }

    public static async Task<NpgsqlDataSourceBuilder> UseEntraAuthenticationAsync(this NpgsqlDataSourceBuilder dataSourceBuilder, TokenCredential? credential = default, CancellationToken cancellationToken = default)
    {
        credential ??= new DefaultAzureCredential();

        if (dataSourceBuilder.ConnectionStringBuilder.Username == null)
        {
            var token = await credential.GetTokenAsync(s_azureDBForPostgresTokenRequestContext, cancellationToken).ConfigureAwait(false);
            SetUsernameFromToken(dataSourceBuilder, token.Token);
        }

        SetPasswordProvider(dataSourceBuilder, credential, s_azureDBForPostgresTokenRequestContext);

        return dataSourceBuilder;
    }

    private static void SetPasswordProvider(NpgsqlDataSourceBuilder dataSourceBuilder, TokenCredential credential, TokenRequestContext tokenRequestContext)
    {
        dataSourceBuilder.UsePasswordProvider(_ =>
        {
            var token = credential.GetToken(tokenRequestContext, default);
            return token.Token;
        }, async (_, ct) =>
        {
            var token = await credential.GetTokenAsync(tokenRequestContext, ct).ConfigureAwait(false);
            return token.Token;
        });
    }

    private static void SetUsernameFromToken(NpgsqlDataSourceBuilder dataSourceBuilder, string token)
    {
        var username = TryGetUsernameFromToken(token);

        if (username != null)
        {
            dataSourceBuilder.ConnectionStringBuilder.Username = username;
        }
        else
        {
            throw new Exception("Could not determine username from token claims");
        }
    }

    private static string? TryGetUsernameFromToken(string jwtToken)
    {
        // Split the token into its parts (Header, Payload, Signature)
        var tokenParts = jwtToken.Split('.');
        if (tokenParts.Length != 3)
        {
            return null;
        }

        // The payload is the second part, Base64Url encoded
        var payload = tokenParts[1];

        // Add padding if necessary
        payload = AddBase64Padding(payload);

        // Decode the payload from Base64Url
        var decodedBytes = Convert.FromBase64String(payload);
        var decodedPayload = Encoding.UTF8.GetString(decodedBytes);

        // Parse the decoded payload as JSON
        var payloadJson = JsonSerializer.Deserialize<JsonElement>(decodedPayload);

        // Try to get the username from 'upn', 'preferred_username', or 'unique_name' claims
        if (payloadJson.TryGetProperty("upn", out var upn))
        {
            return upn.GetString();
        }
        else if (payloadJson.TryGetProperty("preferred_username", out var preferredUsername))
        {
            return preferredUsername.GetString();
        }
        else if (payloadJson.TryGetProperty("unique_name", out var uniqueName))
        {
            return uniqueName.GetString();
        }

        return null;
    }

    private static string AddBase64Padding(string base64) => (base64.Length % 4) switch
    {
        2 => base64 + "==",
        3 => base64 + "=",
        _ => base64,
    };
}