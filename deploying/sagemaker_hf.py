# References:
# https://www.philschmid.de/sagemaker-llama3
# https://huggingface.co/docs/text-generation-inference/messages_api

import boto3
import sagemaker
from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel


class SM:
    def __init__(self, hf_token):
        # Define Model and Endpoint configuration parameter
        self.config = {
            'HF_MODEL_ID': "meta-llama/Meta-Llama-3-8B-Instruct",  # model_id from hf.co/models
            'SM_NUM_GPUS': "1",  # Number of GPU used per replica
            'MAX_INPUT_LENGTH': "2048",  # Max length of input text
            'MAX_TOTAL_TOKENS': "4096",  # Max length of the generation (including input text)
            'MAX_BATCH_TOTAL_TOKENS': "8192",
            # Limits the number of tokens that can be processed in parallel during the generation
            'MESSAGES_API_ENABLED': "true",  # Enable the messages API
            'HUGGING_FACE_HUB_TOKEN': hf_token
        }
        self.instance_type = "(ml.)g5.2xlarge"

    def get_sm_hf_image(self):
        sess = sagemaker.Session()
        sagemaker_session_bucket = None
        if sagemaker_session_bucket is None and sess is not None:
            sagemaker_session_bucket = sess.default_bucket()

        try:
            role = sagemaker.get_execution_role()
        except ValueError:
            iam = boto3.client('iam')
            self.role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

        sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

        print(f"sagemaker role arn: {role}")
        print(f"sagemaker session region: {sess.boto_region_name}")

        # llm image uri
        self.llm_image = get_huggingface_llm_image_uri(
            "huggingface",
            version="2.0.0"
        )

        # ecr image uri
        print(f"llm image uri: {self.llm_image}")

    def get_llm(self):
        # create HuggingFaceModel with the image uri
        self.llm_model = HuggingFaceModel(
            role=self.role,
            image_uri=self.llm_image,
            env=self.config
        )

    def deploy(self):
        # Deploy model to an endpoint
        # https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#sagemaker.model.Model.deploy
        self.llm = self.llm_model.deploy(
            initial_instance_count=1,
            instance_type=self.instance_type,
            container_startup_health_check_timeout=900,  # 10 minutes to be able to load the model
        )

    def infer(self):
        # Prompt to generate
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is deep learning?"}
        ]

        # Generation arguments
        parameters = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "top_p": 0.6,
            "temperature": 0.9,
            "max_tokens": 512,
            "stop": ["<|eot_id|>"],
        }
        chat = self.llm.predict({"messages": messages, **parameters})
        print(chat["choices"][0]["message"]["content"].strip())


if __name__=="__main__":
    sm = SM(hf_token="")
    sm.get_sm_hf_image()
    sm.get_llm()
    sm.deploy()
    sm.infer()