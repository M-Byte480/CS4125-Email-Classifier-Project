from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class BaseModel(ABC):
    def __init__(self) -> None:
        ...


    @abstractmethod
    def train(self, X, y) -> None:
        """
        Train the model using ML Models for Multi-class and mult-label classification.
        :params: df is essential, others are model specific
        :return: classifier
        """
        ...

    @abstractmethod
    def predict(self, X) -> int:
        """

        """
        ...

    @abstractmethod
    def data_transform(self, Z) -> None:
        return

    # def build(self, values) -> BaseModel:
    def build(self, values={}):
        values = values if isinstance(values, dict) else utils.string2any(values)
        self.__dict__.update(self.defaults)
        self.__dict__.update(values)
        return self
