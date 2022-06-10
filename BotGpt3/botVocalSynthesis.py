import glob
import time
import uuid
from gtts import gTTS
import os


def create_bot_vocal_synthesis(save_folder_path, reply):
    for filePath in glob.glob(os.path.join(save_folder_path, '*.mp3')):
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)

    random_mp3_filename = str(uuid.uuid4()) + ".mp3"
    mp3_file_path = os.path.join(save_folder_path, random_mp3_filename)
    myobj = gTTS(text=reply, lang="fr", slow=False)
    myobj.save(mp3_file_path)

    return random_mp3_filename