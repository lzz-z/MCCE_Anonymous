from yue.authentication import get_azure_creds

try:
    from amlt.vault import SubscriptionsCache

    amlt_in_env = True
except ModuleNotFoundError:
    amlt_in_env = False


# mappings for common subscriptions to avoid querying Azure
NAME_TO_ID: dict[str, str] = {
    "Molecular Dynamics": "3eaeebff-de6e-4e20-9473-24de9ca067dc",
}

ID_TO_NAME: dict[str, str] = {
    "3eaeebff-de6e-4e20-9473-24de9ca067dc": "Molecular Dynamics",
}


def get_subscription_id_from_name(subscription_name: str) -> str:
    """
    Retrieves the Azure subscription ID based on the provided subscription name.
    :param subscription_name: The Azure subscription name.
    :return: The subscription ID.
    """

    # Fast path to support the default case without having to query Azure.
    if subscription_name in NAME_TO_ID.keys():
        return NAME_TO_ID[subscription_name]

    from azure.mgmt.resource.subscriptions import SubscriptionClient

    cred = get_azure_creds()
    sub_client = SubscriptionClient(cred)
    subs = [sub for sub in sub_client.subscriptions.list() if sub.display_name == subscription_name]
    if not subs:
        raise RuntimeError(
            f"You do not have access to subscription {subscription_name} (yet). Run `amlt login clear` and try again."
        )
    return subs[0].subscription_id


def get_subscription_name_from_id(subscription_id: str) -> str:
    """
    Retrieves the Azure subscription name based on the provided subscription ID.
    :param subscription_id: The Azure subscription ID.
    :return: The subscription name.
    """

    # Fast path to support the default case without having to query Azure.
    if subscription_id in ID_TO_NAME.keys():
        return ID_TO_NAME[subscription_id]

    from azure.mgmt.resource.subscriptions import SubscriptionClient

    if amlt_in_env:
        # attempt to retrieve subscription name from AMLT cache
        subscriptions_cache = SubscriptionsCache()

        try:
            subscription_name = subscriptions_cache.get_name(subscription_id)
        # TODO: Patch for amlt<10.0, remove when updated
        except Exception:
            for sub in subscriptions_cache.subscriptions:
                if sub["id"] == subscription_id:
                    subscription_name = sub["name"]
                    break
            if "subscription_name" not in locals():
                subscription_name = subscription_id

        # check that the subscription was found in the cache
        if subscription_name != subscription_id:
            print(f"Subscription name is {subscription_name}")
            return subscription_name

    # fall back to Azure SubscriptionClient
    print("Falling back to Azure SubscriptionClient")
    try:
        subscription_client = SubscriptionClient(credential=get_azure_creds())
        subscription = subscription_client.subscriptions.get(subscription_id)
        return subscription.display_name
    except Exception:
        raise RuntimeError(f"Error retrieving name for subscription ID: {subscription_id}")
