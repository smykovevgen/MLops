import os
import tempfile
from typing import Dict, List

import joblib
import numpy as np
from catboost import CatBoostClassifier
from clearml import InputModel, OutputModel, Task, TaskTypes
from sklearn.ensemble import RandomForestClassifier

from exceptions import (AlreadyExistsError, 
                        ConnectionError,
                        InvalidData, 
                        NameKeyError,
                        ParamsTypeError)

MODELS_BUCKET_NAME = "models"
CLEARML_PROJECT_NAME = "nazvanie"


class ModelFactory(object):
    """
    Class for working with models

    """

    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(ModelFactory, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.__available_model_types = {
            "cb": CatBoostClassifier,
            "rf": RandomForestClassifier,
        }
        self.__names_fitted_models: List[str] = []
        self.__models: Dict[Model] = []

    def reload_models(self):
        self.__names_fitted_models: List[str] = []
        # self.__models: Dict[Model] = Storage.reload_models_from_s3()
        if len(self.__models.keys()) != 0:
            for model in self.__models.values():
                if model.fiited:
                    self.__names_fitted_models.append(model.user_model_name)

    def get_available_model_types(self, show: bool = False):
        """
        Getting available model types

        Parameters
        ----------
        show : bool, optional
            True for representation, else False, by default False

        Returns
        -------
        Dict
            If show==True return dict for representaion, else return dict
            with available model classes
        """
        if show:
            return [
                {"model_name": key, "model_type": str(value)}
                for (key, value) in self.__available_model_types.items()
            ]
        else:
            return self.__available_model_types

    def get_models(
        self,
        only_fitted: bool = False,
        all_params: bool = False,
        name_models: str | None = None,
    ):
        """
        Getting all user models

        Parameters
        ----------
        only_fitted : bool, optional
            True if you want to get only fitted model(s), else False,
            by default False
        all_params : bool, optional
            True if you want to get all params, else False,
            by default False
        name_models : str | None, optional
            User's model name of only ONE particular model you want to get,
            by default None

        Returns
        -------
        Dict
            Dictionary with user model name, type of model, params dict,
            fitting bool inidactor

        Raises
        ------
        NameKeyError
            Occurs if the model with the same name was not found
            or was not fitted
        """
        self.reload_models()
        if name_models is not None and only_fitted:
            if name_models in self.__names_fitted_models:
                name_models = [name_models]
            else:
                raise NameKeyError(
                    "A model with the same name was not found or was not fitted"
                )
        elif name_models is not None:
            if name_models in list(self.__models.keys()):
                name_models = [name_models]
            else:
                raise NameKeyError("There is no model with this name")
        else:
            if only_fitted:
                name_models = self.__names_fitted_models
            else:
                name_models = list(self.__models.keys())
        return [
            {
                "user_model_name": user_model_name,
                "type_model": self.__models[user_model_name].type_model,
                "params": self.get_params(user_model_name, all_params),
                "fitted": self.__models[user_model_name].fiited,
            }
            for user_model_name in name_models
        ]

    def init_new_model(
        self, type_model: str, user_model_name: str, params: dict = {}
    ):
        """
        Initialize one model and store it in memory

        Parameters
        ----------
        type_model : str
            Shortname of base model type
        user_model_name : str
            User's model name
        params : dict, optional
            User's params for model, by default {}

        Returns
        -------
        Dict
            Dictionary with user model name, type of model, params dict,
            fitting bool inidactor

        Raises
        ------
        AlreadyExistsError
            Occurs if a model with same name already exists

        NameKeyError
            Occurs if there is an error in model type or model name
        """
        self.reload_models()
        if user_model_name in self.__models.keys():
            raise AlreadyExistsError(
                "A model with the same name already exists"
            )
        if type_model not in self.__available_model_types.keys():
            raise NameKeyError(
                "The selected model is not in the list of available ones"
            )
        self.__models[user_model_name] = Model(
            self.__available_model_types[type_model],
            type_model,
            user_model_name,
            params=params,
        )

        return {
            "user_model_name": user_model_name,
            "type_model": type_model,
            "params": params,
            "fitted": False,
        }

    def model_fit(self, X: np.array, y: np.array, user_model_name: str):
        """
        Model fitting

        Parameters
        ----------
        X : np.array
            Training data
        y : np.array
            Target data
        user_model_name : str
            Name of the model to be fitted

        Raises
        ------
        NameKeyError
            Occurs if a model with the same name was not found
        """
        try:
            self.reload_models()
            self.__models[user_model_name].fit(X, y)
            self.__names_fitted_models.append(user_model_name)
        except KeyError:
            raise NameKeyError("There is no model with this name")

    def model_predict(self, X: np.array, user_model_name: str):
        """
        Making prediction on data

        Parameters
        ----------
        X : np.array
            Data to predict on
        user_model_name : str
            Name of the model to use for prediction

        Returns
        -------
        np.array
            Predictions

        Raises
        ------
        NameKeyError
            Occurs if a model with the same name was not found
            or was not fitted
        """
        self.reload_models()
        if user_model_name in self.__names_fitted_models:
            preds = self.__models[user_model_name].predict(X)
            return preds
        else:
            raise NameKeyError(
                "A model with the same name was not found or was not fitted"
            )

    def get_params(self, user_model_name: str, all: bool = False) -> dict:
        """
        Getting params of model by name

        Parameters
        ----------
        user_model_name : str
            User model name
        all : bool, optional
            True if you want to get all params, False if you want to get
            params setted by user, by default False

        Returns
        -------
        dict
            Dict with params
        """
        self.reload_models()
        return self.__models[user_model_name].get_params(all)

    def delete_model(self, user_model_name: str):
        """
        Deleting model by name

        Parameters
        ----------
        user_model_name : str
            Model name to delete

        Raises
        ------
        NameKeyError
            Occurs if there is no model with same name
        """
        try:
            self.reload_models()
            del self.__models[user_model_name]
            if user_model_name in self.__names_fitted_models:
                self.__names_fitted_models.remove(user_model_name)
        except KeyError:
            raise NameKeyError("There is no model with this name")


class Model:
    """
    Class describing one model

    Raises
    ------
    ParamsTypeError
        Error im params
    InvalidData
        Error in data
    """

    getting_params_func_names = {
        CatBoostClassifier: CatBoostClassifier.get_all_params,
        RandomForestClassifier: RandomForestClassifier.get_params,
    }

    def __init__(
        self, base_model, type_model: str, user_model_name: str, params: Dict
    ) -> None:
        """
        Constructs all the necessary attributes for the Model object.

        Parameters
        ----------
        base_model : _type_
            Base model class
        type_model : str
            Shortname for base model
        user_model_name : str
            User-specified model name
        params : Dict
            Params for model

        Raises
        ------
        ParamsTypeError
            Error in params
        """
        self.type_model: str = type_model
        self.params: Dict = params
        self.user_model_name: str = user_model_name
        self.fiited: bool = False
        try:
            self.base_model = base_model(**self.params)
        except TypeError:
            raise ParamsTypeError("Incorrect model hyperparameters passed")

    def fit(self, X: np.array, y: np.array):
        """
        Fit one model

        Parameters
        ----------
        X : np.array
            Train data
        y : np.array
            Target data

        Raises
        ------
        InvalidData
            Error in data
        """
        try:
            task = Task.init(
                project_name=CLEARML_PROJECT_NAME,
                task_name=self.user_model_name,
                tags=[self.type_model, "fit"],
            )
            task.set_parameters(self.params)
            model = self.base_model.__class__(**self.params)
            model.fit(X=X, y=y)
            joblib.dump(model, f"{self.user_model_name}.pkl")
            self.fiited = True
            self.base_model = model
            task.close()
        except:
            task.close()
            raise InvalidData("Incorrect training data")

    def predict(self, X: np.array):
        """
        Return prediction on X

        Parameters
        ----------
        X : np.array
            Data to predict on

        Raises
        ------
        InvalidData
            Error in data
        """
        try:
            task = Task.init(
                project_name=CLEARML_PROJECT_NAME,
                task_name=self.user_model_name,
                tags=[self.type_model, "predict"],
                task_type=TaskTypes.inference,
            )
            input_model = InputModel(
                project=CLEARML_PROJECT_NAME,
                name=self.user_model_name + " - " + self.user_model_name,
            )
            task.connect(input_model)
            model = self.base_model
            preds = model.predict(X)
            task.set_parameters(self.params)
            self.base_model = model
            task.close()
            return preds
        except:
            task.close()
            raise InvalidData("Incorrect data for prediction")

    def get_params(self, all: bool = False) -> dict:
        """
        Getting params of model

        Parameters
        ----------
        all : bool, optional
            True if you want to get all params, False if you want to get
            params setted by user, by default False

        Returns
        -------
        dict
            Params dict
        """
        if all is True:
            return self.getting_params_func_names[type(self.base_model)](
                self.base_model
            )
        elif all is False:
            return self.params
