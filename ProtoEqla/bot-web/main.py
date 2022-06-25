from flask import Flask, render_template, request, session
from PIL import Image
from ProtoEqla.lib.ocr_task import OcrTask
from chatbot import chatbot_query
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'any-random-string'
demo_files = {
    "IMG_6077.jpg":"/home/ionut/Code/OSS/HITW/ProtoEqla/bot-web/static/data/IMG_6077.jpg",
    "IMG_6078.jpg":"/home/ionut/Code/OSS/HITW/ProtoEqla/bot-web/static/data/IMG_6078.jpg",
    "IMG_6079.jpg":"/home/ionut/Code/OSS/HITW/ProtoEqla/bot-web/static/data/IMG_6079.jpg",
    "exemple_contrat.png":"/home/ionut/Code/OSS/HITW/ProtoEqla/bot-web/static/data/exemple_contrat.png"
}
examples_contexts = {}

def __build_context(example):
    doc_text = "\n".join(examples_contexts[example])

    with open('botContext.txt') as f:
        text = f.read()

    text = text.replace("{CONTENT_DOC}", doc_text) + "\nMoi:" + "\nSalut" + "\nIris:"
    print(text)
    return text


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def home_post():
    incoming_msg = request.get_data().decode().split("|")
    question = incoming_msg[0]
    example = incoming_msg[1]
    has_changed = bool(int(incoming_msg[2]))
    if has_changed:
        session.clear()
        session['chat_logs'] = __build_context(example)

    chat_logs = session.get('chat_logs')
    chatbot_reply = chatbot_query(question, chat_logs, os.path.join(app.root_path, 'static'))
    session['chat_logs'] = chatbot_reply.full
    return f"{chatbot_reply.reply}###{chatbot_reply.audio_filename}"


if __name__ == '__main__':
    ocr_task = OcrTask()
    for key, fp in demo_files.items():
        img = Image.open(fp)
        pred = ocr_task.predict(img)
        txt = ocr_task.get_text(pred)
        examples_contexts[key] = txt
    PORT = 8082
    app.run(port=PORT)

