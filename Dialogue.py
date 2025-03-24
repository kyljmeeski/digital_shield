from typing import List


class Dialogue:
    def __init__(self, src: str):
        self.__source = src

    def sentences(self) -> List[str]:
        for sentence in self.__source.replace('!', '.').replace('?', '.').split('.'):
            yield sentence.strip()
