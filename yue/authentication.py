import os
from functools import lru_cache

import azure.identity as identity
import azure.identity.aio as identity_aio


class CachedChainedTokenCredential(identity.ChainedTokenCredential):
    # DefaultAzureCredential implements the below snippet already, but ChainedTokenCredential does not.
    # ChainedTokenCredential does not implement it to avoid a breaking change
    # due to custom credentials, which we're not concerned about.
    def get_token(self, *args, **kwargs):
        if self._successful_credential:
            return self._successful_credential.get_token(*args, **kwargs)
        return super().get_token(*args, **kwargs)


class AIOCachedChainedTokenCredential(identity_aio.ChainedTokenCredential):
    # DefaultAzureCredential implements the below snippet already, but ChainedTokenCredential does not.
    # ChainedTokenCredential does not implement it to avoid a breaking change
    # due to custom credentials, which we're not concerned about.
    async def get_token(self, *args, **kwargs):
        if self._successful_credential:
            return await self._successful_credential.get_token(*args, **kwargs)
        return await super().get_token(*args, **kwargs)


identity.CachedChainedTokenCredential = CachedChainedTokenCredential
identity_aio.CachedChainedTokenCredential = AIOCachedChainedTokenCredential


@lru_cache(maxsize=1)
def get_azure_creds():
    return _get_chained_token_credential(identity)


@lru_cache(maxsize=1)
def get_azure_creds_async():
    return _get_chained_token_credential(identity_aio)


def _get_chained_token_credential(identity_impl):
    """
    Configures a ChainedTokenCredential that tries a couple of authentication methods in turns.

    Args:
        identity_impl: Identity library, either azure.identity or azure.identity.aio.
    """
    if identity_impl not in (identity, identity_aio):
        raise ValueError(
            "identity_impl is expected to be either azure.identity or azure.identity.aio."
        )

    # It is important that we try AzureCliCredential first, as we want to avoid picking up
    # managed identities that may exist but are unusable (e.g. on GCR sandboxes).
    # Because we cannot control the order of the credentials in DefaultAzureCredential, we
    # use a ChainedTokenCredential to ensure that AzureCliCredential is tried first.
    # We're taking a conservative approach whereby we are only allowing credentials we
    # know we need. For now these are:
    # - WorkloadIdentityCredential.
    # - ManagedIdentityCredential

    # NOTE: unlike DefaultAzureCredential, the get_token method of ChainedTokenCredential does
    # not leverage the _successful_credential attribute. We should consider subclassing
    # ChainedTokenCredential to add this functionality.

    chained_credentials = [
        identity_impl.AzureCliCredential(),
        identity_impl.DefaultAzureCredential(
            exclude_environment_credential=True,
            exclude_cli_credential=True,
            exclude_shared_token_cache_credential=True,
            exclude_developer_cli_credential=True,
            exclude_powershell_credential=True,
            exclude_interactive_browser_credential=True,
            exclude_visual_studio_code_credentials=True,
            # DEFAULT_IDENTITY_CLIENT_ID is a variable exposed in Azure ML Compute jobs that
            # has the client id of the user-assigned managed identity in it.
            # See https://learn.microsoft.com/en-us/azure/machine-learning/how-to-identity-based-service-authentication?view=azureml-api-2&tabs=cli#compute-cluster
            # In case it's not set the ManagedIdentityCredential will default to using
            # the system-assigned managed identity.
            managed_identity_client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID"),
        ),
    ]

    if os.getenv("USE_INTERACTIVE_BROWSER_CREDENTIAL") == "true":
        # USE_INTERACTIVE_BROWSER_CREDENTIAL is a variable used by aml-endpoint-manager
        # to enable interactive browser credentials during the local deployment.
        print("Enabling authentication via interactive browser credentials...")

        if os.getenv("INTERACTIVE_BROWSER_CREDENTIAL_open_browser") == "false":
            # To allow interactive browser credentials to be used by a docker container,
            # identity_impl._credentials.browser._open_browser function is monkey pathced
            # so that it prints out the authentication link rather than opening a browser window.
            print(
                "Configuring interactive browser credentials to not open the browser automatically."
            )

            def _updated_open_browser(url: str) -> bool:
                print("Please navigate to this URL to authenticate:", url)
                return True

            identity_impl._credentials.browser._open_browser = _updated_open_browser

        interactive_browser_credential = identity.InteractiveBrowserCredential(
            redirect_uri=os.getenv("INTERACTIVE_BROWSER_CREDENTIAL_redirect_uri"),
            additionally_allowed_tenants=["72f988bf-86f1-41af-91ab-2d7cd011db47"],
        )

        class AuthCodeRedirectServerForDockerContainer(
            interactive_browser_credential._server_class
        ):
            """
            A class to  override hostname parameter so that requests can be processed
            authentication server created inside the docker container.
            """

            def __init__(self, hostname, port, timeout):
                super().__init__(hostname="0.0.0.0", port=port, timeout=timeout)

        interactive_browser_credential._server_class = AuthCodeRedirectServerForDockerContainer

        chained_credentials.insert(0, interactive_browser_credential)

    return identity_impl.ChainedTokenCredential(*chained_credentials)


if __name__ == "__main__":
    get_azure_creds()
