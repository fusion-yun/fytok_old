
from spdm.util.AttributeTree import AttributeTree


class Identifier(AttributeTree):
    def __init__(self, *args, name=None, index=0, description=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name or ""
        self.index = index
        self.description = ""
