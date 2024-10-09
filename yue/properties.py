from typing import Any, Dict, Generic, List, Literal, Optional, Union
from datetime import datetime
from typeguard import check_type

import pint
from yue.base_document import BunnetDocument
from pydantic import AnyUrl, BaseModel, Field, field_validator
from uuid import UUID, uuid4

#from earthshots.orfb.calculations.schemas.filetypes import SMILES

pint_registry = pint.UnitRegistry()

Property = Literal["logS", "reduction_potential", "oxidation_potential"]

MLModelIdentifier = Literal[
    "logS_endpoint", "oxidation_potential_endpoint", "reduction_potential_endpoint"
]


class PropertyModel(BaseModel):
    """
    This is intended to store property:value pairs and associated metadata.
    """

    name: Property = Field(..., help="The name of the property.")

    value: Any = Field(..., help="The value for the property.")

    units: str = Field(..., help="The units in which the property value is expressed.")

    method: Literal["experiment", "dft", "ml"] = Field(
        ..., help="The method by which the property value was determined."
    )

    origin: Union[str, AnyUrl, MLModelIdentifier] = Field(
        ...,
        help="Information related to the determination. If method is `ml`, origin is MLModelIdentifier. If method is `experiment`, "
        + "origin is AnyURL. If method is `dft`, origin is the UUID of calculation",
    )

    created_on: Optional[datetime] = Field(
        None,
        help="Date and time of document creation. Automatically generated.",
        validate_default=True,
    )

    @field_validator("units")
    @classmethod
    def check_units(cls, value):
        try:
            pint_registry(value)
        except pint.errors.UndefinedUnitError:
            raise (f"The unit {value} was not recognized by pint.")
        return value

    @field_validator("origin")
    @classmethod
    def check_origin(cls, value, info):
        if info.data["method"] == "ml":
            assert check_type(
                value, MLModelIdentifier
            ), "When method originates from ML, origin must be a MLModelInfo."
        elif info.data["method"] == "experiment":
            assert isinstance(
                value, AnyUrl
            ), "When method originates from experiment, origin must be AnyUrl."
        elif info.data["method"] == "dft":
            try:
                UUID(value)
            except ValueError:
                raise ValueError("When method originates from dft, origin must be UUID.")
        else:
            raise ValueError("Unexpected value for method.")
        return value

    @field_validator("created_on")
    @classmethod
    def check_created_on(cls, value) -> datetime:
        """
        Automatically document the date and time of document creation.
        """
        if isinstance(value, datetime):
            return value
        else:
            return datetime.utcnow()


class PropertiesModel(BaseModel ): # ,Generic[SMILES]
    """
    top-level model holding multiple property values associated with a SmilesDoc
    """

    uuid: str = Field(
        default_factory=lambda: str(uuid4()), help="Unique identifer for the document."
    )

    smiles:  str = Field(
        ...,
        help="The parent SMILES.",
    )

    properties: Dict[Property, List[PropertyModel]] = Field(
        default_factory=dict,
        help="Property values and associated metadata for the structure, accessible using a Property literal key.",
    )

    @field_validator("uuid")
    @classmethod
    def check_uuid(cls, value):
        try:
            UUID(value)
            return value
        except ValueError:
            raise ValueError("Invalid UUID provided to field uuid.")


class PropertiesDoc(PropertiesModel, BunnetDocument):
    """
    Mixed-in class with bunnet functionality.
    """

    pass
