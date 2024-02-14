from interpreter import interpreter

interpreter.offline = True # Disables online features like Open Procedures
interpreter.llm.model = "openai/x" # Tells OI to send messages in OpenAI's format
interpreter.llm.api_key = "fake_key" # LiteLLM, which we use to talk to LM Studio, requires this
interpreter.llm.api_base = "http://192.168.1.106:1234/v1" # Point this at any OpenAI compatible server
interpreter.llm.context_window = "20480"
interpreter.llm.max_tokens = "8000"

interpreter.chat()
