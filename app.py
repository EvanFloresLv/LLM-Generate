from llm.settings import ModelSettings, GeminiCredentials

print("Initializing config...")
config = ModelSettings()
user = GeminiCredentials()

print(user._credentials)
print(user._project)