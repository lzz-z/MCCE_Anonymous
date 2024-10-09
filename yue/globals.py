import logging

try:
    from azure.quantum import Workspace

    DEFAULT_QUANTUM_WORKSPACE_RESOURCE_ID = "/subscriptions/3eaeebff-de6e-4e20-9473-24de9ca067dc/resourceGroups/quantum-test-ws-rg/providers/Microsoft.Quantum/Workspaces/ai4squantumtest0prod"
    DEFAULT_QUANTUM_WORKSPACE_LOCATION = "eastus"
    DEFAULT_QUANTUM_WORKSPACE = Workspace(
        resource_id=DEFAULT_QUANTUM_WORKSPACE_RESOURCE_ID,
        location=DEFAULT_QUANTUM_WORKSPACE_LOCATION,
    )
except ModuleNotFoundError:
    logging.info("azure-quantum not found in environment, DEFAULT_QUANTUM_WORKSPACE not available.")

EARTHSHOTS_AZURE_SUBSCRIPTION_ID = "3eaeebff-de6e-4e20-9473-24de9ca067dc"
EARTHSHOTS_AZURE_SUBSCRIPTION_NAME = "Molecular Dynamics"
EARTHSHOTS_AZURE_RESOURCE_GROUP = "sfm-rg"
EARTHSHOTS_AML_KEYVAULT_NAME = "a4s-sfm-keyvault"
EARTHSHOTS_AML_RESOURCE_GROUP = "sfm-ws-rg"
EARTHSHOTS_AML_WORKSPACE = "sfm-ws"
EARTHSHOTS_COSMOS_DB_ACCOUNT = "earthshots-ru"

ORFB_AMULET_STORAGE_CONTAINER_NAME = "earthshots-orfb-test"
ORFB_STORAGE_ACCOUNT_NAME = "orfb0eastus"
ORFB_MONGO_DATABASE_NAME = "orfb"
ORFB_AZURE_CACHE_ZINC22_SMILESDOC_NAME = "earthshots-orfb-earthshots2-ru-zinc22-smilesdocs"
