const submitButton = document.getElementById('submitButton');
const chatbotInput = document.getElementById('chatbotInput');
const chatbotOutput = document.getElementById('chatbotOutput');
const audioSpan = document.getElementById('audioSpan');

// Submit + Reply
submitButton.onclick = userSubmitEventHandler;
chatbotInput.onkeyup = userSubmitEventHandler;

function userSubmitEventHandler(event) {
    if (
        (event.keyCode && event.keyCode === 13) ||
        event.type === 'click'
    ) {
        chatbotOutput.innerText = 'réfléchi...';
        askChatBot(chatbotInput.value);
    }
}

function askChatBot(userInput) {
    const myRequest = new Request('/', {
        method: 'POST',
        body: userInput
    });

    fetch(myRequest).then(function (response) {
        if (!response.ok) {
            throw new Error('HTTP error, status = ' + response.status);
        } else {
            return response.text();
        }
    }).then(function (text) {
        let values = text.split("###")
        console.log(values)
        chatbotInput.value = '';
        chatbotOutput.innerText = values[0];

        let audio = document.createElement('audio');
        let audioSource = document.createElement('source');
        audioSource.setAttribute('type', 'audio/mp3');
        audioSource.setAttribute('src', '/static/' + values[1]);
        audioSource.setAttribute('autoplay', '');
        audio.appendChild(audioSource)
        audioSpan.appendChild(audio);
        audio.load();
        audio.play();

    }).catch((err) => {
        console.error(err);
    });
}


// STT
navigator.mediaDevices.getUserMedia({audio: true})
window.SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.interimResults = true;
recognition.lang = 'fr-FR';

recognition.addEventListener('result', event => {
    const transcript = event.results[0][0].transcript;
    if (event.results[0].isFinal) {
        chatbotInput.value = transcript + "?";
    }
});

recognition.addEventListener('end', recognition.start);
recognition.start();