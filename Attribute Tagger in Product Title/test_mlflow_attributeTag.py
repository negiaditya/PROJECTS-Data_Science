import mlflow.pyfunc
import pandas as pd
import json

test = True
model_path = "attribute_tagger_v-1-0-0"

if test==True:
    loaded_model = mlflow.pyfunc.load_model(model_path)
    print('loaded_model from model_path=', model_path)

    input_file_path = 'test_text1.json'

    if 'json' in input_file_path:
        input_df = pd.read_json(path_or_buf=input_file_path,
                                       dtype={'text': 'str'},
                                       orient='split')

    print(loaded_model.predict(input_df))


