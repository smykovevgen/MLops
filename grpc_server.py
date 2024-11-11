from concurrent import futures

import models_fitting_pb2
import models_fitting_pb2_grpc
import numpy as np

import grpc
from rest.model_training import ModelFactory


class Greeter(models_fitting_pb2_grpc.ModelTransferServicer):
    def GetAvModelTypes(self, request, context):
        return models_fitting_pb2.AvailableModelTypesResponse(
            list_of_models=allm.get_available_model_types(show=request.show)
        )

    def InitModel(self, request, context):
        dict_init = allm.init_new_model(
            type_model=request.type_model,
            user_model_name=request.user_model_name,
            params=dict(request.params_string) | dict(request.params_float),
        )
        return models_fitting_pb2.InitModelResponse(
            user_model_name=dict_init["user_model_name"],
            type_model=dict_init["type_model"],
        )

    def FitModel(self, request, context):
        X = np.array(request.X)
        X = X.reshape(-1, int(len(X) / len(list(request.y))))
        y = np.array(request.y)
        try:
            allm.model_fit(
                np.array(X),
                np.array(y),
                request.user_model_name,
            )
            status = "Model fitted"
        except:
            status = "Error"
        return models_fitting_pb2.FitModelResponse(status=status)

    def PredModel(self, request, context):
        X = np.array(request.X)
        X = X.reshape(-1, request.n_feats)
        y = allm.model_predict(X=X, user_model_name=request.user_model_name)
        y = y.flatten()
        return models_fitting_pb2.PredictModelResponse(y=y)

    def DelModel(self, request, context):
        print(request.user_model_name)

        try:
            allm.delete_model(request.user_model_name)
            status = f"Model {request.user_model_name} deleted"
        except:
            status = "Error"
        return models_fitting_pb2.DeleteModelResponse(status=status)


def serve():
    port = "50051"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    models_fitting_pb2_grpc.add_ModelTransferServicer_to_server(
        Greeter(), server
    )
    server.add_insecure_port("localhost:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


allm = ModelFactory()
serve()
