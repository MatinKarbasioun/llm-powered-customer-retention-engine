from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    OPENAI_API_KEY: str 
    KAGGLE_USERNAME: str
    KAGGLE_KEY: str
    HF_TOKEN: str
    domains: set[str] = set()

    more_settings: SubModel = SubModel()

    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8") 
    
    def conf(SettingsConfigDict):
        
    