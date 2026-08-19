"""Provider-neutral vector-store error taxonomy."""


class VectorStoreError(RuntimeError):
    """Base class for vector-store failures."""


class VectorStoreUnavailableError(VectorStoreError):
    """The configured vector store cannot be reached."""


class VectorCollectionConfigurationError(VectorStoreError):
    """The configured collection is missing or incompatible."""


class VectorDimensionMismatchError(VectorCollectionConfigurationError):
    """Stored and requested embedding dimensions do not match."""


class VectorUpsertError(VectorStoreError):
    """A vector-store upsert failed."""


class VectorRetrievalError(VectorStoreError):
    """A vector-store query failed or returned malformed data."""


class MissingClientFilterError(VectorRetrievalError):
    """A scoped query was attempted without a usable client identity."""
