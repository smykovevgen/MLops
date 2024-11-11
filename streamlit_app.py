import json

import pandas as pd
import requests
import streamlit as st


def streamlit_app():
    st.title("Ml")
    app_mode = st.sidebar.selectbox(
        "What to do",
        [
            "Get available model types",
            "Get models",
            "Fit model",
            "Get model prediction",
        ],
    )

    def get_available_model_types():
        return requests.get(
            "http://0.0.0.0:8000/get_available_model_types"
        ).json()

    def get_models():
        return requests.get("http://0.0.0.0:8000/get_models").json()

    if app_mode == "Get available model types":
        res = get_available_model_types()

        print(res)
        st.table(res)

    if app_mode == "Get models":
        res = get_models()
        av_mod_types = get_available_model_types()
        st.table(res)

        type_model = st.selectbox(
            "Model type", list(d["model_name"] for d in av_mod_types)
        )
        user_model_name = st.text_input("Model name", placeholder="cb1")
        params = st.text_input(
            "Hyperparams in json format", placeholder="{'random_state': 43}"
        )
        # params = json.loads('{"random_state": 43}')
        if st.button("Init new model"):
            requests.post(
                f"http://0.0.0.0:8000/init_new_model?type_model={type_model}&user_model_name={user_model_name}",
                json=dict(params),
            )
            st.write(type_model, user_model_name)

    if app_mode == "Fit model":
        models = get_models()
        model_name = st.selectbox(
            "Model name", list(d["user_model_name"] for d in models)
        )
        train_x = {}
        target_col = st.text_input("Target columns", placeholder="y")
        file = st.file_uploader("train_data in csv")
        if file is not None:
            df = pd.read_csv(file, sep=";")
            train_values = df[target_col].values.tolist()
            df_x = df.drop(columns=target_col, inplace=False, axis=1)
            for col in df_x.columns:
                train_x[col] = df_x[col].values.tolist()
            data = {}
            data["X"] = train_x
            data["y"] = train_values
            json_data = json.dumps(data)

            print(json_data)
            result = requests.put(
                f"http://0.0.0.0:8000/model_fit/{model_name}", json=data
            ).status_code
            print(result)
            if result == 200:
                st.success("Model fitted")

    if app_mode == "Get model prediction":
        models = get_models()
        model_name = st.selectbox(
            "Model name", list(d["user_model_name"] for d in models)
        )
        test_x = {}
        target_col = st.text_input(
            "Target true columns (optional)", placeholder="y"
        )
        file = st.file_uploader("train_data in csv")
        if file is not None:
            df = pd.read_csv(file, sep=";")
            if target_col != "":
                train_values = df[target_col].values.tolist()
                df_x = df.drop(columns=target_col, inplace=False, axis=1)
            else:
                df_x = df
                train_values = []
            for col in df_x.columns:
                test_x[col] = df_x[col].values.tolist()
            data = {}
            data["X"] = test_x
            data["y"] = train_values
            json_data = json.dumps(data)
            result = requests.put(
                f"http://0.0.0.0:8000/model_predict/{model_name}", json=data
            ).json()
            df["preds"] = result["preds"]
            st.table(df)


streamlit_app()
