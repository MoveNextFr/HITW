import openai
import os

from botVocalSynthesis import create_bot_vocal_synthesis

openai.api_key = os.getenv("API_KEY")
movebot_start_sequence = "\nMovebot : "
user_start_sequence = "\nMoi : "


class ChatbotReply:
    def __init__(self, context, reply, audio_filename):
        reply = reply.replace("\n", "")
        self.full = context + reply
        self.reply = reply
        self.audio_filename = audio_filename


def get_default_context() -> str:
    with open('botContext.txt') as f:
        lines = f.readlines()

    return '\n '.join(lines)


def chatbot_query(query, context, static_folder_path) -> ChatbotReply:
    if context == "" or context is None:
        query_with_context = get_default_context() + user_start_sequence + query + movebot_start_sequence
    else:
        query_with_context = context + user_start_sequence + query + movebot_start_sequence
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=query_with_context,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    reply = str(response['choices'][0]['text'])
    mp3_filename = create_bot_vocal_synthesis(static_folder_path, reply)
    return ChatbotReply(query_with_context, reply, mp3_filename)
