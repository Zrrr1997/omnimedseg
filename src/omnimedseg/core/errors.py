"""Exceptions raised by the OmniMedSeg pipeline."""


class OmniMedSegError(Exception):
    """Base class for all OmniMedSeg errors."""


class DownloadError(OmniMedSegError):
    """A dataset download produced no usable files."""


class ManualDownloadRequired(OmniMedSegError):
    """The dataset cannot be downloaded automatically; the user has to fetch it."""


class ConverterNotFoundError(OmniMedSegError, LookupError):
    """No converter function exists for the requested dataset."""


class PipelineError(OmniMedSegError):
    """One or more datasets failed to process. `failures` maps dataset name -> exception."""

    def __init__(self, failures):
        self.failures = failures
        names = ", ".join(failures)
        super().__init__(f"{len(failures)} dataset(s) failed: {names}")
