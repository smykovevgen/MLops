from typing import Dict

import models_fitting_pb2
import models_fitting_pb2_grpc
import numpy as np

import grpc



def get_model_types(stub, show: bool = True):
    av_model_types = stub.GetAvModelTypes(
        models_fitting_pb2.AvailableModelTypesRequest(show=True)
    )
    return list(av_model_types.list_of_models)


def init_model(
    stub,
    type_model: str,
    user_model_name: str,
    params: Dict = {"random_state": 42, "n_estimators": 100},
):
    params_string = {k: v for k, v in params.items() if isinstance(v, str)}
    params_float = {
        k: v
        for k, v in params.items()
        if isinstance(v, float) or isinstance(v, int)
    }
    model_init = stub.InitModel(
        models_fitting_pb2.InitModelRequest(
            type_model=type_model,
            user_model_name=user_model_name,
            params_string=params_string,
            params_float=params_float,
        )
    )
    print(
        "Модель создана: ", model_init.user_model_name, model_init.type_model
    )


def fit_model(stub, X: np.array, y: np.array, user_model_name: str):
    status = stub.FitModel(
        models_fitting_pb2.FitModelRequest(
            X=np.array(X).flatten(), y=y, user_model_name=user_model_name
        )
    )
    print(status)


def pred_model(stub, X: np.array, user_model_name: str):
    n_feats = int(np.array(X).shape[0])
    preds = stub.PredModel(
        models_fitting_pb2.PredictModelRequest(
            X=np.array(X).flatten(),
            n_feats=n_feats,
            user_model_name=user_model_name,
        )
    ).y
    return preds


def del_model(stub, user_model_name: str):
    status = stub.DelModel(
        models_fitting_pb2.DeleteModelRequest(user_model_name=user_model_name)
    ).status
    return status


def run():
    print("client started")
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = models_fitting_pb2_grpc.ModelTransferStub(channel)

        av_model_types = get_model_types(stub, show=True)
        print("Доступные типы моделей: ", av_model_types)

        init_model(
            stub,
            type_model="cb",
            user_model_name="cb1",
            params={"random_state": 42, "n_estimators": 101},
        )
        init_model(
            stub,
            type_model="cb",
            user_model_name="cb2",
            params={"random_state": 43, "n_estimators": 102},
        )
        init_model(
            stub,
            type_model="rf",
            user_model_name="rf1",
            params={"random_state": 44, "n_estimators": 103},
        )
        fit_model(
            stub,
            X=np.array([[2, 3, 7], [4, 5, 6]]),
            y=np.array([4, 5, 2]),
            user_model_name="cb1",
        )
        model_preds_array = pred_model(
            stub, X=[[2, 3, 7], [4, 5, 6]], user_model_name="cb1"
        )
        print("Model_preds: ", model_preds_array)

        status = del_model(stub, user_model_name="cb1")
        print(status)


run()
