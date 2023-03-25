1. "attribute_tagger_v-1-0-0" is the module which predicts color,size,brand from a given product title. It is a mlflow.pyfunc module. 

2. "data_final.csv" is the output dataframe which predicted over "product_title.json.tar (1).gz".

3. "test_mlflow_attributeTag.py" tests over a single test case using pyfunc module.

4. "test_restapi_attributeTag.ipynb" tests over a single test case using restapi.

5. "test_text1.json" is the input test case and created manually as a json input payload.

6. Ignore "Problem2_attributeTag.ipynb", it has some of the data analysis and model building.

7. To create a rest api, Run the following command in cmd/conda shell:
	'''mlflow models serve -m attribute_tagger_v-1-0-0  -p 5000'''
   Make sure attribute_tagger_v-1-0-0 is in the current directory and port is 5000.

8. To dockerise the above module:
	'''mlflow models build-docker -m attribute_tagger_v-1-0-0 -n "attr_tag"'''
	'''docker run -p 5001:8080 "attr_tag"'''
   
   'attr_tag' is the image name and is being exposed at port 5000.


