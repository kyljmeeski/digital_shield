from __future__ import annotations

from numpy import ndarray
from sentence_transformers import SentenceTransformer
from torch import Tensor


class Phrase:
    def __init__(self, txt: str, mdl: SentenceTransformer):
        self.__text = txt
        self.__model = mdl

    def vector(self) -> list[Tensor] | ndarray | Tensor:
        return self.__model.encode(self.__text)

    def __str__(self) -> str:
        return self.__text
