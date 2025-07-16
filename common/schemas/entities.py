from dataclasses import asdict, dataclass
from typing import List


@dataclass
class SampleRequestPayload:
    sample_field: str
