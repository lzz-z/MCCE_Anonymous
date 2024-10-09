from datetime import datetime
from functools import partial
from typing import Any, Mapping, Union

from bunnet import Document as _Document
from bunnet.exceptions import CollectionWasNotInitialized
from bunnet.odm.settings.document import DocumentSettings
from monty.json import MSONable, jsanitize
from pydantic import BaseModel, Field, validator

# This matches `jsanitize` used by jobflow library, this allows conversion
# several additional datatypes into JSON automatically, such as
# pymatgen Molecule objects (or any "MSONable" type). The dependency on `monty`
# is not strictly required for `ai4s-data`, but it makes the class more useful
# for reuse elsewhere.
_jsanitize = partial(jsanitize, strict=True, allow_bson=True, enum_values=True)


class BunnetDocument(_Document):
    """
    A Document class defined by bunnet, built upon pydantic
    BaseModel.

    This class also allows Python objects that subclass
    MSONable (monty) to be encoded automatically. This is
    likely only relevant for some users, but is a harmless
    addition to the class to those that do not need it.

    Several default fields are also defined (@module, @class,
    and @version) which act as pointers to the specific version
    of the document model used to create a document. It also defines
    a created_at field, since Mongo/Cosmos does not store
    the time at which a document is created automatically.
    """

    @classmethod
    def get_settings(cls) -> DocumentSettings:
        """
        Get document settings, which was created on
        the initialization step. We override this
        function here to ensure MSONable objects
        can be serialized without having to manually
        specify bson_encoders in every BunnetDocument's
        Settings.

        :return: DocumentSettings class
        """
        if cls._document_settings is None:
            raise CollectionWasNotInitialized
        settings = cls._document_settings

        # patch default settings to encode common data types used by us
        settings.bson_encoders[MSONable] = _jsanitize
        # this is necessary for nested models which also include MSONable types
        settings.bson_encoders[BaseModel] = _jsanitize

        return settings

    # storing "@module", "@class" and "@version" as fields is a convention
    # when using "MSONable" objects (via the `monty`` package), this allows
    # a class to be automatically reconstructed from a dictionary
    # or JSON object, as well as tracking the specific version of the code
    # that was uesd to generate the object

    # these fields are set automatically by the validators below
    doc_module: str = Field("", alias="@module")
    doc_class: str = Field("", alias="@class")
    doc_version: str = Field("", alias="@version")

    # a generally-useful field, Mongo/Cosmos does not store this by default
    # TODO: revisit later, this stored as str due to an issue with storing datetime directly,
    # possibly a conflict with monty
    created_at: str = Field(
        default_factory=lambda: str(datetime.now()),
        help="Specifies the time at which this document was created.",
    )

    @validator("doc_module", always=True, pre=True)
    def _set_module(cls, v) -> str:
        return cls.__module__

    @validator("doc_class", always=True, pre=True)
    def _set_class(cls, v) -> str:
        return cls.__name__

    @validator("doc_version", always=True, pre=True)
    def _set_version(cls, v) -> str:
        # since feynman doesn't define a good __version__,
        # use the current git commit hash instead
        return "null"

    @classmethod
    def count_documents(cls, *args: Union[Mapping[str, Any], bool]) -> int:
        """
        Performance of .countDocuments() on Cosmos is quite poor,
        often resulting in timeouts. This method is a workaround.

        Args:
            query: query in bunnet Document.find() syntax

        Returns:
            A count of documents matching query
        """
        query = cls.find(*args).get_filter_query()
        count = cls.get_motor_collection().database.command(
            {"count": cls.get_collection_name(), "query": query}
        )
        if "n" in count:
            return count["n"]
        else:
            # fallback for when not using Cosmos, or during testing
            return cls.find(*args).count()
