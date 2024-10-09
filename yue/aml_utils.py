import logging
import os
from dataclasses import dataclass
from functools import lru_cache

try:
    from azureml.core import Run, Workspace
    from azureml.exceptions import RunEnvironmentException

    aml_in_env = True
except ModuleNotFoundError:
    aml_in_env = False

logger = logging.getLogger(__name__)

class NotRunningInAMLError(Exception):
    pass

from typing import Union

@dataclass(frozen=True)
class AMLRunInfo:
    #id: str | None = None
    id: Union[str, None] = None
    portal_url: str = ""

    @staticmethod
    def from_run() -> "AMLRunInfo":
        try:
            from msrest.exceptions import ClientException
        except ModuleNotFoundError:
            return AMLRunInfo()

        try:
            run = get_run()
        except NotRunningInAMLError:
            return AMLRunInfo()
        except ClientException as e:
            logger.info(f"While trying to obtain AML run, got exception: {e}")
            # As a fallback, get the portal URL from environment variable
            return AMLRunInfo(portal_url=os.environ.get("AML_RUN_PORTAL_URL", ""))
        return AMLRunInfo(run.id, run.get_portal_url())


@lru_cache(maxsize=1)
def get_run() -> "Run":
    """Gets the current AML run."""
    if not aml_in_env:
        raise NotRunningInAMLError("azureml-core not found.")

    try:
        # This will only succeed in a process running in Azure ML
        return Run.get_context(allow_offline=False)
    except RunEnvironmentException as e:
        raise NotRunningInAMLError("Not running in AML.") from e


def is_running_in_aml() -> bool:
    """Returns True if the process is running in AML."""
    try:
        get_run()
        return True
    except NotRunningInAMLError:
        return False


def get_secret_from_workspace(secret_name: str, workspace: "Workspace | None" = None) -> str:
    """Gets a secret from the default keyvault of a workspace.

    Args:
        secret_name: The name of the secret to retrieve.
        workspace: (optional) The workspace to retrieve the secret from. If None,
            the workspace of the current AML run is used.

    Returns:
        The secret value.

    Raises:
        NotRunningInAMLError: If not running in Azure ML or AML SDK is not installed.
    """
    if workspace is None:
        workspace = get_run().experiment.workspace
    kv = workspace.get_default_keyvault()
    return kv.get_secret(secret_name)
