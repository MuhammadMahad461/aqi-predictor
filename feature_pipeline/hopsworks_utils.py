import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

def get_feature_store():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )
    return project.get_feature_store()