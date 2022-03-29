from .GeoObject import GeoObject


class Surface(GeoObject):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def points(self, *args, **kwargs):
        return super().points(*args, **kwargs)

    def map(self, u, *args, **kwargs):
        return NotImplemented

    def derivative(self, u, *args, **kwargs):
        return NotImplemented

    def dl(self, u, *args, **kwargs):
        return NotImplemented

    def pullback(self, func, *args, **kwargs):
        return NotImplemented

    def make_one_form(self, func):
        return NotImplemented
