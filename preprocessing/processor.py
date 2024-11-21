import numpy as np
import pandas as pd
import stanza
from stanza.pipeline.core import DownloadMethod
from transformers import pipeline
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Config import Config


class DataProcessor:
    PATH_TO_APP = "data/AppGallery.csv"
    PATH_TO_PURCHASES = "data/Purchasing.csv"
    PATH_TO_APP_PREPROCESSED = "data/AppGalleryPreprocessed.csv"
    PATH_TO_PURCHASES_PREPROCESSED = "data/PurchasingPreprocessed.csv"
    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=4, max_df=0.90)

    @staticmethod
    def load_data(file_path):
        """Loads the data from the specified location"""
        df = pd.read_csv(file_path)
        return df

    @staticmethod
    def save_data(file_path, df):
        """Saves the data to a specified location"""
        df.to_csv(file_path, index=False)

    @staticmethod
    def renaming_cols(data_frame: pd.DataFrame):
        df = data_frame
        # convert the dtype object to unicode string
        df[Config.TICKET_SUMMARY] = df[Config.TICKET_SUMMARY].values.astype('U')
        df[Config.INTERACTION_CONTENT] = df[Config.INTERACTION_CONTENT].values.astype('U')

        # Optional: rename variable names for remembering easily
        df.rename(columns={
            Config.INTERACTION_CONTENT: "x_ic",
            Config.TICKET_SUMMARY: "x_ts",
            Config.GROUPED: "y1",
            Config.CLASS_COL: "y2",
            Config.TYPE_COLS[0]: "y3",
            Config.TYPE_COLS[1]: "y4",
        }, inplace=True)

        return df

    @staticmethod
    def de_duplication(data_frame):
        # Remove all rows with duplicates
        df_no_duplicates = data_frame[~data_frame.duplicated(subset="Interaction id", keep=False)]

        print(data_frame.shape[0] - df_no_duplicates.shape[0], "Rows removed due to duplicates and incorrect labelling")
        return df_no_duplicates

    @staticmethod
    def replace_nan_data_in_column(data_frame, column_name):
        for idx, entry in enumerate(data_frame[column_name]):
            if entry == 'nan' or entry is None or entry != entry:
                data_frame.loc[idx, column_name]= ""
        return data_frame

    @staticmethod
    def remove_noise(data_frame):
        ### Step 4: Noise Removal
        # remove re:
        # remove extrac white space
        # remove
        noise = "(sv\\s*:)|(wg\\s*:)|(ynt\\s*:)|(fw(d)?\\s*:)|(r\\s*:)|(re\\s*:)|(\\[|\\])|(aspiegel support issue submit)|(null)|(nan)|((bonus place my )?support.pt 自动回复:)"
        data_frame["x_ts"] = data_frame["x_ts"].str.lower().replace(noise, " ", regex=True).replace(r'\\s+', ' ', regex=True).str.strip()

        data_frame["x_ic"] = data_frame["x_ic"].str.lower()
        noise_1 = [
            "(from :)|(subject :)|(sent :)|(r\\s*:)|(re\\s*:)",
            "(january|february|march|april|may|june|july|august|september|october|november|december)",
            "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
            "(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
            "\\d{2}(:|.)\\d{2}",
            "(xxxxx@xxxx\\.com)|(\\*{5}\\([a-z]+\\))",
            "dear ((customer)|(user))",
            "dear",
            "(hello)|(hallo)|(hi )|(hi there)",
            "good morning",
            "thank you for your patience ((during (our)? investigation)|(and cooperation))?",
            "thank you for contacting us",
            "thank you for your availability",
            "thank you for providing us this information",
            "thank you for contacting",
            "thank you for reaching us (back)?",
            "thank you for patience",
            "thank you for (your)? reply",
            "thank you for (your)? response",
            "thank you for (your)? cooperation",
            "thank you for providing us with more information",
            "thank you very kindly",
            "thank you( very much)?",
            "i would like to follow up on the case you raised on the date",
            "i will do my very best to assist you"
            "in order to give you the best solution",
            "could you please clarify your request with following information:"
            "in this matter",
            "we hope you(( are)|('re)) doing ((fine)|(well))",
            "i would like to follow up on the case you raised on",
            "we apologize for the inconvenience",
            "sent from my huawei (cell )?phone",
            "original message",
            "customer support team",
            "(aspiegel )?se is a company incorporated under the laws of ireland with its headquarters in dublin, ireland.",
            "(aspiegel )?se is the provider of huawei mobile services to huawei and honor device owners in",
            "canada, australia, new zealand and other countries",
            "\\d+",
            "[^0-9a-zA-Z]+",
            "(\\s|^).(\\s|$)"]
        for noise in noise_1:
            # print(noise)
            data_frame["x_ic"] = data_frame["x_ic"].replace(noise, " ", regex=True)
        data_frame["x_ic"] = data_frame["x_ic"].replace(r'\\s+', ' ', regex=True).str.strip()

        return data_frame

    def vectorize_data(self, data_frame):
        ## Step 6: Textual data numerically:
        x_ic = self.tfidfconverter.fit_transform(data_frame["x_ic"]).toarray()
        x_ts = self.tfidfconverter.transform(data_frame["x_ts"]).toarray()
        X = np.concatenate((x_ic, x_ts), axis=1)
        # remove bad test cases from test dataset
        # convert the 4 labels in to an array of labels
        y = {
            "y1": data_frame["y1"].to_numpy(),
            "y2": data_frame["y2"].to_numpy(),
            "y3": data_frame["y3"].to_numpy(),
            "y4": data_frame["y4"].to_numpy()
        }
        return X, y

    def vectorize_unclassified_data(self, data_frame):
        x_ic = self.tfidfconverter.transform(data_frame["x_ic"]).toarray()
        x_ts = self.tfidfconverter.transform(data_frame["x_ts"]).toarray()
        X = np.concatenate((x_ic, x_ts), axis=1)
        return X

    @staticmethod
    def remove_nan_rows(X, y):
        indices_of_nans = []
        for idx, y_val in enumerate(y):
            if y_val == 'nan' or y_val != y_val:
                indices_of_nans.append(idx)

        X = np.delete(X, indices_of_nans, axis=0)
        y = np.delete(y, indices_of_nans)
        return X, y

    def split_and_balance_data(X, y, test_size=0.2, min_class_samples=3):
        y_series = pd.Series(y)
        good_y_value = y_series.value_counts()[y_series.value_counts() >= min_class_samples].index
        X_good, y_good = X[y_series.isin(good_y_value)], y[y_series.isin(good_y_value)]
        X_bad, y_bad = X[~y_series.isin(good_y_value)], y[~y_series.isin(good_y_value)]

        adjusted_test_size = test_size * len(X) / len(X_good)
        X_train, X_test, y_train, y_test = train_test_split(X_good, y_good, test_size=adjusted_test_size,
                                                            random_state=0)
        X_train = np.concatenate((X_train, X_bad), axis=0)
        y_train = np.concatenate((y_train, y_bad), axis=0)

        return X_train, X_test, y_train, y_test

    def train(self, classifier_model, data):
        X_train, y_train, X_test, y_test = data
        ### Step 10: Model selection for classification
        classifier_model = RandomForestClassifier(n_estimators=1000, random_state=0)

        ### Step 11: Model Training
        classifier_model.fit(X_train, y_train)

        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

        y_pred = classifier_model.predict(X_test)

        p_result = pd.DataFrame(classifier_model.predict_proba(X_test))
        p_result.columns = classifier_model.classes_
        print(p_result)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    @staticmethod
    def translate_data_frame(data_frame):
        data_frame["x_ic"] = DataProcessor.trans_to_en(data_frame["x_ic"].to_list())
        data_frame["x_ts"] = DataProcessor.trans_to_en(data_frame["x_ts"].to_list())
        return data_frame

    # Translation
    @staticmethod
    def trans_to_en(texts):
        t2t_m = "facebook/m2m100_418M"
        t2t_pipe = pipeline(task='text2text-generation', model=t2t_m)

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


        return text_en_l
