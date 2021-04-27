from typing import TypeVar,  Sequence
import collections

if __name__ == "__main__":

    T=TypeVar('T')

    class List(Sequence[T]):
        def __init__(self, d=None, *args,   **kwargs):
            self._data = d or []

        def __getitem__(self,k:int)->T:
            return self._data[k]

        def __len__(self)->int:
            return len(self._data)

    L=List[int] 
    l=L([1,2,3,4])

    print(isinstance(l,collections.abc.Sequence))