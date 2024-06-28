import os
import getpass
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic  # Assuming this is the correct import for the Claude model
from langchain_google_genai import ChatGoogleGenerativeAI

class BaseModel:

    def __init__(self, model_name, temperature=0):
        self.model_name = model_name
        self.model_temperature = temperature
        self.api_key = self.get_api_key()
        self.initialize_model()

    def get_api_key(self):
        api_key_env = self.api_key_env_name()
        if api_key_env not in os.environ:
            os.environ[api_key_env] = getpass.getpass(f"Enter {api_key_env}: ")
        return os.environ[api_key_env]

    def api_key_env_name(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def initialize_model(self):
        raise NotImplementedError("Subclasses should implement this method.")

class OpenAIModel(BaseModel):

    def __init__(self, model_name= "gpt-3.5-turbo", temperature=0):
        super().__init__(model_name, temperature)

    def api_key_env_name(self,model_name = "gpt-3.5-turbo"):
        return "OPENAI_API_KEY"

    def initialize_model(self):
        self.model = ChatOpenAI(model=self.model_name, temperature=self.model_temperature, openai_api_key=self.api_key)

    def get_model(self):
        return self.model
    
class GroqModel(BaseModel):

    def __init__(self, model_name= "llama3-70b-8192", temperature=0):
        super().__init__(model_name, temperature)

    def api_key_env_name(self):
        return "GROQ_KEY"

    def initialize_model(self):
        self.model = ChatGroq(model=self.model_name, temperature=self.model_temperature, api_key=self.api_key)

    def get_model(self):
        return self.model
    
class ClaudeModel(BaseModel):
    def __init__(self, model_name="claude-3-sonnet-20240229", temperature=0):
        super().__init__(model_name, temperature)

    def api_key_env_name(self):
        return "ANTHROPIC_API_KEY"

    def initialize_model(self):
        self.model = ChatAnthropic(model=self.model_name, temperature=self.model_temperature, anthropic_api_key=self.api_key)
    
    def get_model(self):
        return self.model
        
class GoogleModel(BaseModel):

    def __init__(self, model_name= "gemini-1.5-flash", temperature=0):
        super().__init__(model_name, temperature)

    def api_key_env_name(self,model_name = "gpt-3.5-turbo"):
        return "GOOGLE_API_KEY"

    def initialize_model(self):
        self.model = ChatGoogleGenerativeAI(model=self.model_name, temperature=self.model_temperature, google_api_key=self.api_key)

    def get_model(self):
        return self.model

