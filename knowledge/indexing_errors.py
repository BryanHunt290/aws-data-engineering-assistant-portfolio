"""Typed, sanitized failures for production vector indexing."""


class IndexingRuntimeError(RuntimeError):
    """Base class for safe indexing-runtime failures."""


class IndexingConfigurationError(IndexingRuntimeError, ValueError):
    """Raised when indexing configuration is incomplete or inconsistent."""


class IndexingSecretError(IndexingRuntimeError):
    """Raised when vector-store credentials cannot be resolved safely."""


class IndexingSecretSchemaError(IndexingSecretError):
    """Raised when a resolved secret does not match the expected schema."""


class IndexingDependencyError(IndexingRuntimeError):
    """Raised when optional production runtime dependencies are unavailable."""


class IndexingBatchLimitError(IndexingRuntimeError):
    """Raised before work exceeds a configured invocation bound."""


class ManifestWriteConflictError(IndexingRuntimeError):
    """Raised when optimistic manifest reconciliation is exhausted."""


class RedriveSafetyError(IndexingRuntimeError):
    """Raised when a requested redrive would violate scope or safety rules."""


class IntegrationValidationError(IndexingRuntimeError):
    """Raised when an integration validation would be unsafe or ambiguous."""
