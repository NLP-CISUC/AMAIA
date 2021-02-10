from flask import Flask, render_template, request
import controler
import ast

from agents.BertAgent import BertAgent
from agents.Word2vecAgent import Word2vecAgent

'''
Flask app controler, responsible for back-end of the app
'''

app = Flask(__name__)

@app.route('/')
def home():
    #Page rendering
    return render_template("home.html")

#Logic for chatbot interface
@app.route("/get")
def get_bot_response():
    #Getting user input
    input_text = request.args.get('msg')

    #Using controler to return the most suitable response
    return str(controler.web_chat_interface(input_text))


def load_config(config_file):
    lines = {}
    with open(config_file,"r") as f:
        contents = f.read()
        lines = ast.literal_eval(contents)
        f.close()
    return lines


if __name__ == '__main__':
    #Loading models and starting app
    configs = load_config("config.txt")

    controler.start_controler(
        configs["agents"], configs["corpus"], configs["out_of_domain_corpus"],
        configs["out_of_domain_classifier"], configs["number_of_answers_per_agent"], configs["decision"],
        configs["theta"], configs["w2v_embeddings"], configs["bert_embeddings"])

    app.run(host="localhost", port=5001, debug=True)
