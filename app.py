from flask import Flask, render_template, request
from llama_index import SimpleDirectoryReader, LlamaListIndex, LlamaSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain.chat_models import ChatOpenAI
app = Flask(__name__)

# Define the index and chatbot logic
class IndexAndChatbot:
    def __init__(self, directory_path):
        self.directory_path = directory_path
        self.index = self.construct_index()
    
    def construct_index(self):
        max_input_size = 4096
        num_outputs = 512
        max_chunk_overlap = 20
        chunk_size_limit = 600

        prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=num_outputs))

        documents = SimpleDirectoryReader(self.directory_path).load_data()

        index = GPTVectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

        index.save_to_disk('index.json')

        return index

    def chatbot(self, input_text):
        input_text += "Don't provide information outside the given context."
        response = self.index.query(input_text, response_mode="default")
        return response.response

# Initialize your chatbot with the data directory path
chatbot_instance = IndexAndChatbot("data")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    input_text = request.form['user_input']
    response = chatbot_instance.chatbot(input_text)
    return render_template('index.html', user_input=input_text, chatbot_response=response)

if __name__ == '__main__':
    app.run(debug=True)
