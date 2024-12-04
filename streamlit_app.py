import json
from src.feature_engineering import FeatureEngineer, labelEncoding 
import logging

import numpy as np
import pandas as pd
import re
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main


def main():
    st.title("End to End Machine Predictive Maintenance Classification Pipeline with ZenML")

    # whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")

    st.markdown(
        """ 
    #### Problem Statement 
     The objective here is to predict the machine failure prediction for a given machine based on features like Air temperature [k], Process temperature [k], Torque [Nm], etc. I will be using [ZenML](https://zenml.io/) to build a production-ready pipeline to predict machine failure prediction for the next machine failure.    """
    )
    # st.image(whole_pipeline_image, caption="Whole Pipeline")
    project_gif = "_assets/project_architecture.gif"
    st.image(project_gif, caption="Project Architecture")

    st.markdown(
        """ 
    Above is a figure of the whole pipeline, we first ingest the data, preprocess it, train the model, and evaluate the model, and if data source changes or any hyperparameter values changes, deployment will be triggered, and (re) trains the model and if the model meets minimum accuracy requirement, the model will be deployed.
    """
    )

    st.markdown(
        """ 
    #### Description of Features 
    This app is designed to predict the machine failure prediction for a given machine. You can input the features of the product listed below and get the machine failure prediction. 
    | Models        | Description   | 
    | ------------- | -     | 
    | UDI | unique identifier ranging from 1 to 10000 | 
    | Product ID | unique identifier for each machine | 
    | Type   | consisting of a letter L , M, or H for low (50% of all products), medium (30%), and high (20%) as product quality variants and a variant-specific serial number |  
    | air temperature [K] |       generated using a random walk process later normalized to a standard deviation of 2 K around 300 K | 
    | process temperature [K] |       generated using a random walk process normalized to a standard deviation of 1 K, added to the air temperature plus 10 K. |
    | rotational speed [rpm] |    calculated from powepower of 2860 W, overlaid with a normally distributed noise  | 
    | torque [Nm] |    torque values are normally distributed around 40 Nm with an Ïƒ = 10 Nm and no negative values. |
    | tool wear [min] |    The quality variants H/M/L add 5/3/2 minutes of tool wear to the used tool in the process. 
    """
    )
    udi = st.sidebar.number_input("Unique Identifier (UID)", min_value=0, step=1, format="%d")  
    # type = st.sidebar.selectbox("Type", options=["Low (1)", "Medium(2)", "High (3)"]) 
    type = st.sidebar.number_input("Type")
    product_id = st.sidebar.number_input("Product ID")
    air_temperature = st.sidebar.number_input("Air Temperature [K]")
    process_temperature = st.sidebar.number_input("Process Temperature [K]")
    rotational_speed = st.sidebar.number_input("Rotational Speed [rpm]")
    torque = st.sidebar.number_input("Torque [Nm]")
    tool_wear = st.sidebar.number_input("Tool Wear [min]")
    # machine_failure = st.sidebar.checkbox("Machine Failure")


    if st.button("Predict"):
        service = prediction_service_loader(
        pipeline_name="continuous_deployment_pipeline",
        pipeline_step_name="mlflow_model_deployer_step",
        running=False,
        )
        if service is None:
            st.write(
                "No service could be found. The pipeline will be run first to create a service."
            )
            #  run_main()

        # df = pd.read_csv('/mnt/c/Users/HP/ml_projects/predictive_maintenance_mlops/extracted_data/predictive_maintenance.csv')
        # df = df.sample(n=100)
        # label_encoding = labelEncoding(features=["Type", "Product ID", "Failure Type"])
        # encoded_data = FeatureEngineer(label_encoding)
        # df = encoded_data.apply_feature_engineering(df)
        # df.drop(['Target', 'Failure Type_encoded'], axis=1, inplace=True)
        # df.columns = [re.sub(r"[<>[\]]", "", col) for col in df.columns]
        # df = pd.DataFrame(
        #     {
        #         "UDI": [101, 102, 103],
        #         "Air temperature K": [295.5, 299.0, 300.5],
        #         "Process temperature K": [310.2, 307.8, 315.0],
        #         "Rotational speed rpm": [106.0, 107.0, 110.0],
        #         "Torque Nm": [308.5, 310.0, 320.0],
        #         "Tool wear min": [320.0, 315.0, 330.0],
        #         "Type_encoded": [110.0, 111.0, 112.0],
        #         "Product ID_encoded": [315.0, 320.0, 325.0]
        #     }
        # )
        df = pd.DataFrame(
            {
                "UDI": [udi],
                "Air temperature K": [air_temperature],
                "Process temperature K": [process_temperature],
                "Rotational speed rpm": [rotational_speed],
                "Torque Nm": [torque],
                "Tool wear min": [tool_wear],
                "Type_encoded": [type],
                "Product ID_encoded": [product_id]
            }
        )
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        logging.info(f"Payload for prediction: {data}")
        pred = service.predict(data)
        print("Prediction", pred)

        if isinstance(pred, np.ndarray):
            pred = pred[0]

        failure = None 
        if pred == 0:
            failure = "No Failure"
        elif pred == 1:
            failure = 'Heat Dissipation Failure' 
        elif pred == 2:
            failure = 'Power Failure'
        elif pred == 3:
            failure = 'Overstrain Failure'
        elif pred == 4:
            failure ='Tool Wear Failure'
        else:
            failure = 'Random Failures' 

        st.success(
            f"Your Machine failure prediction is: {failure}"
        )
    else:
        st.warning("Please enter the relevant information to predict.")

if __name__ == "__main__":
    main()