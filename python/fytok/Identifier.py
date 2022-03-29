from dataclasses import dataclass

@dataclass
class Identifier:
    name: str = "unnamed"
    index: int = 0
    description: str = ""