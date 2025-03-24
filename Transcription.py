import re
from typing import List

from transformers import pipeline
from transformers import AutoTokenizer


class Transcription:
    def __init__(self, src: str, mdl: str = 'RUPunct/RUPunct_big'):  # there are `medium' and 'big' versions as well
        self.__source = src
        self.__tokenizer = AutoTokenizer.from_pretrained(mdl, strip_accents=False, add_prefix_space=True)
        self.__classifier = pipeline('ner', model=mdl, tokenizer=self.__tokenizer, aggregation_strategy="first")

    def sentences(self) -> List[str]:
        text = ''
        for item in self.__classifier(self.__source):
            text += " " + self.__process_token(item['word'].strip(), item['entity_group'])
        phrases = re.split(r"[.!?]+", text)
        return [phrase.lower().strip() for phrase in phrases]

    @staticmethod
    def __process_token(token, label) -> str:  # are all of these signs necessary?
        # if label == "LOWER_PERIOD":
        #     return token + "."
        # if label == "UPPER_PERIOD":
        #     return token.capitalize() + "."
        # if label == "UPPER_TOTAL_PERIOD":
        #     return token.upper() + "."
        if label in ["LOWER_PERIOD", "UPPER_PERIOD", "UPPER_TOTAL_PERIOD"]:
            return token + "."

        # if label == "LOWER_QUESTION":
        #     return token + "?"
        # if label == "UPPER_QUESTION":
        #     return token.capitalize() + "?"
        # if label == "UPPER_TOTAL_QUESTION":
        #     return token.upper() + "?"
        if label in ["LOWER_QUESTION", "UPPER_TOTAL_QUESTION", "UPPER_TOTAL_PERIOD"]:
            return token + "?"

        # if label == "LOWER_VOSKL":
        #     return token + "!"
        # if label == "UPPER_VOSKL":
        #     return token.capitalize() + "!"
        # if label == "UPPER_TOTAL_VOSKL":
        #     return token.upper() + "!"
        if label in ["LOWER_VOSKL", "UPPER_VOSKL", "UPPER_TOTAL_VOSKL"]:
            return token + "!"

        # if label == "LOWER_QUESTIONVOSKL":
        #     return token + "?!"
        # if label == "UPPER_QUESTIONVOSKL":
        #     return token.capitalize() + "?!"
        # if label == "UPPER_TOTAL_QUESTIONVOSKL":
        #     return token.upper() + "?!"
        if label in ["LOWER_QUESTIONVOSKL", "UPPER_QUESTIONVOSKL", "UPPER_TOTAL_QUESTIONVOSKL"]:
            return token + "?"

        # if label == "LOWER_MNOGOTOCHIE":
        #     return token + "…"
        # if label == "UPPER_MNOGOTOCHIE":
        #     return token.capitalize() + "…"
        # if label == "UPPER_TOTAL_MNOGOTOCHIE":
        #     return token.upper() + "…"
        if label in ["LOWER_MNOGOTOCHIE", "UPPER_MNOGOTOCHIE", "UPPER_TOTAL_MNOGOTOCHIE"]:
            return token + "."

        # if label == "LOWER_PERIODCOMMA":
        #     return token + ";"
        # if label == "UPPER_PERIODCOMMA":
        #     return token.capitalize() + ";"
        # if label == "UPPER_TOTAL_PERIODCOMMA":
        #     return token.upper() + ";"
        if label in ["LOWER_PERIODCOMMA", "UPPER_PERIODCOMMA", "UPPER_TOTAL_PERIODCOMMA"]:
            return token + "."

        return token

        # if label == "LOWER_COMMA":
        #     return token + ","
        # if label == "UPPER_COMMA":
        #     return token.capitalize() + ","
        # if label == "UPPER_TOTAL_COMMA":
        #     return token.upper() + ","

        # if label == "LOWER_DEFIS":
        #     return token + "-"
        # if label == "UPPER_DEFIS":
        #     return token.capitalize() + "-"
        # if label == "UPPER_TOTAL_DEFIS":
        #     return token.upper() + "-"

        # if label == "LOWER_TIRE":
        #     return token + "—"
        # if label == "UPPER_TIRE":
        #     return token.capitalize() + " —"
        # if label == "UPPER_TOTAL_TIRE":
        #     return token.upper() + " —"

        # if label == "LOWER_DVOETOCHIE":
        #     return token + ":"
        # if label == "UPPER_DVOETOCHIE":
        #     return token.capitalize() + ":"
        # if label == "UPPER_TOTAL_DVOETOCHIE":
        #     return token.upper() + ":"

        # if label == "UPPER_O":
        #     return token.capitalize()
        # if label == "UPPER_TOTAL_O":
        #     return token.upper()
        # if label == "LOWER_O":
        #     return token








