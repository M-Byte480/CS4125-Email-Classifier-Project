import warnings
import numpy as np
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from utilities.logger.error_logger import ErrorLogger
from utilities.logger.indentation_decorator import IndentationDecorator
from utilities.logger.warning_logger import WarningLogger

#The legacy translator code, which we need to adapt wo work with a list-based interface
class OldTranslator:
    @staticmethod
    def trans_to_en(texts : np.array):
        error_logger = IndentationDecorator(ErrorLogger("Translator"), "[trans_to_en]")
        warning_logger = IndentationDecorator(WarningLogger("Translator"), "[trans_to_en]")
        with warnings.catch_warnings(record=True) as caught_warning:

            t2t_m = "facebook/m2m100_418M"

            model = M2M100ForConditionalGeneration.from_pretrained(t2t_m)
            tokenizer = M2M100Tokenizer.from_pretrained(t2t_m)
            nlp_stanza = stanza.Pipeline(lang="multilingual",
                                         processors="langid",
                                         download_method=DownloadMethod.REUSE_RESOURCES)
            language_map = {
                "fro": "fr",  # Old French
                "la": "it",  # Latin
                "nn": "no",  # Norwegian (Nynorsk)
                "kmr": "tr",  # Kurmanji
                "mt": "pl"   # maltese to polish because there is no maltese (in the dataset or the model)
            }

            text_en_l = []
            for text in texts:
                # Empty strings get appended
                if text == "":
                    text_en_l.append("")
                    continue
                try:
                    doc = nlp_stanza(text)
                    detected_lang = doc.lang

                    # If language is english append
                    if detected_lang == "en":
                        text_en_l.append(text)
                        continue
                    # Detected lang
                    detected_lang = language_map.get(detected_lang, detected_lang)

                    tokenizer.src_lang = detected_lang
                    encoded_hi = tokenizer(text, return_tensors="pt")
                    generated_tokens = model.generate(
                        **encoded_hi,
                        forced_bos_token_id=tokenizer.get_lang_id("en")
                    )
                    text_en = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

                    text_en_l.append(text_en)


                except Exception as e:
                    error_logger.log("Error occured")
                    error_logger.log(str(e))

            for warning in caught_warning:
                if issubclass(warning.category, FutureWarning):
                    warning_logger.log(f"{warning.message}")

        return text_en_l
