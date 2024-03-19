import os
from flask import Flask, render_template, request
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = Flask(__name__)

# load the env file
load_dotenv()

# get the api key from the env file
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
model_id = "hmzkhnswt/tinyllama_customerSupport_hmc"

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForCausalLM.from_pretrained(model_id)
text_generator = pipeline("text-generation", model=model,
                          tokenizer=tokenizer, max_length=1000)


def chat(user_input):
    model_response = text_generator(user_input, max_length=300, temperature=0.7)[
        0]['generated_text']
    return model_response


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form['name_of_input']
        print("User: ", question)
        res = chat(question)
        print("AI: ", res)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
