import mlflow.pyfunc
import pandas as pd
import cloudpickle
import json
import sys
import os
import yaml
import utils
import product_config

class attribute_tagger(mlflow.pyfunc.PythonModel):
    def load_context(self, context):

        from utils import get_model
        
        learning_store_path = context.artifacts["learning_store_path"]
        
        print('attribute_tagger learning_store_path=', learning_store_path)
        self.model = get_model(learning_store_path)
        print('SUCEESS: Done with load_context. Ready for predict')
        

    def predict(self, context, input_df):
        print('\n ------------ Incoming new predict request...')
        print('model_input_df=', input_df)
        
        text = input_df['text'].iloc[0]

        output_fields = utils.get_attributes(text,self.model)

        return [{"input_text":text,"output":output_fields}]

    
with open('conda.yaml','r') as envfile:
#    conda_env = yaml.safe_load(envfile)
    conda_env = yaml.safe_load(envfile.read().replace(u'\x00', ''))

model_path = "output/attribute_tagger_v-1-0-0"

code_path = product_config.code_path
print('Loading src files from code_path=', code_path)


# path for artifacts.
artifacts = {
    "learning_store_path" : "artifacts/best-model.pt"
}

save_model = True
if save_model:
    print('================\n')
    print('mlflow.pyfunc.save_model START!')
    
    mlflow.pyfunc.save_model(path=model_path, 
                             python_model=attribute_tagger(),
                             code_path=code_path,
                             artifacts=artifacts,
                             conda_env=conda_env)

    print('SUCCESS in mlflow.pyfunc.save_model!')
    print('model_path=', model_path)
    print('================')

    
load_model = True
if load_model:
    print('================\n')
    print('mlflow.pyfunc.load_model START!')

    # Evaluate the model
    loaded_model = mlflow.pyfunc.load_model(model_path)
    print('loaded_model from model_path=', model_path)
    
    input_file_path = 'data/text1.json'
    
    if 'json' in input_file_path:
        input_df = pd.read_json(path_or_buf=input_file_path,
                                       dtype={'text': 'str'},
                                       orient='split')
    
    print(loaded_model.predict(input_df))
