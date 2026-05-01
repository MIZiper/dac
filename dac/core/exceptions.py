"""Typed exceptions for the DAC framework."""


class DacError(Exception):
    """Base exception for all DAC-specific errors."""


class NodeNotFoundError(DacError):
    """Raised when a requested data or context node is not found."""


class ActionError(DacError):
    """Base exception for action-related errors."""


class ActionConfigError(ActionError):
    """Raised when an action's configuration is invalid."""


class ActionExecutionError(ActionError):
    """Raised when an action fails during execution."""


class ContainerError(DacError):
    """Base exception for Container-related errors."""


class ClassNotFoundError(ContainerError):
    """Raised when a registered class path cannot be resolved."""


class ContextError(DacError):
    """Base exception for context-related errors."""


class TypeAgencyError(DacError):
    """Raised when a type agency handler fails."""


class ScenarioError(DacError):
    """Raised when scenario loading or parsing fails."""


class SnippetError(DacError):
    """Raised when dynamically executed snippet code fails."""
