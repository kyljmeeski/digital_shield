from __future__ import annotations

from sentence_transformers import SentenceTransformer

from Phrase import Phrase


class Pair:
    def __init__(self, frst: Phrase, scnd: Phrase, mdl: SentenceTransformer):
        self.__first = frst
        self.__second = scnd
        self.__model = mdl

    def similarity(self) -> int | float | bool:
        return self.__model.similarity(self.__first.vector(), self.__second.vector()).item()
