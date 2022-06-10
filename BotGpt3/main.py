from flask import Flask, render_template, request, session
from chatbot import chatbot_query
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any-random-string'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def home_post():
    incoming_msg = request.get_data().decode()
    chat_logs = session.get('chat_logs')

    chatbot_reply = chatbot_query(incoming_msg, chat_logs, os.path.join(app.root_path, 'static'))
    session['chat_logs'] = chatbot_reply.full
    return f"{chatbot_reply.reply}###{chatbot_reply.audio_filename}"


if __name__ == '__main__':
    PORT = 8082
    app.run(port=PORT)