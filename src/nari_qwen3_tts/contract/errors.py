"""Errors shared across the engine and API boundary."""


class SynthesisError(RuntimeError):
    """Base error surfaced by the bounded synthesis engine."""


class ServiceUnavailable(SynthesisError):  # noqa: N818 - public protocol vocabulary
    pass


class ServiceCapacityExceeded(SynthesisError):  # noqa: N818 - public protocol vocabulary
    pass


class BackpressureExceeded(SynthesisError):  # noqa: N818 - public protocol vocabulary
    pass


class RequestCancelled(SynthesisError):  # noqa: N818 - public protocol vocabulary
    pass


class RequestRejected(SynthesisError):  # noqa: N818 - public protocol vocabulary
    """A well-formed request cannot be served by the configured engine."""


class LiveInputClosedError(SynthesisError):
    """A validated live-input update could not publish before Generation ended."""


class StreamingTextControlTokenError(ValueError):
    """Live target text contains a reserved tokenizer control token."""


__all__ = [
    "BackpressureExceeded",
    "LiveInputClosedError",
    "RequestCancelled",
    "RequestRejected",
    "ServiceCapacityExceeded",
    "ServiceUnavailable",
    "SynthesisError",
    "StreamingTextControlTokenError",
]
