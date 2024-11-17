// fetch('https://icanhazdadjoke.com/slack')
//     .then(data => data.json())
//     .then(jokeData => {
//         const jokeText = jokeData.attachments[0].text;
//         const jokeElement = document.getElementById('jokeElement');

//         jokeElement.innerHTML = jokeText;
//     })

// Get references to the input field, chat box, and send button
const chatInput = document.getElementById('chat-input');
const chatBox = document.getElementById('chat-box');
const sendBtn = document.getElementById('send-btn');
const voiceBtn = document.getElementById('voice-btn'); // Button for voice input

// Send message when the user clicks the send button
sendBtn.addEventListener('click', sendMessage);

// Send message when the user presses 'Enter'
chatInput.addEventListener('keypress', function(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
});

// Function to send a message
async function sendMessage() {
    const messageText = chatInput.value.trim();
    if (messageText === '') return;

    // Add user's message to the chat box
    addMessage(messageText, 'user');

    const response = await fetch('http://127.0.0.1:8000/message/', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json', // Set the request content type to JSON
        },
        body: JSON.stringify({ msg: messageText }), // Send the message as JSON
    });
    if (response.ok) {
        const jsonResponse = await response.json();
        addMessage(jsonResponse.message, 'Chatbot'); // Display response
    }

    // Clear the input field
    chatInput.value = '';
}

// Function to add a message to the chat box
function addMessage(text, sender) {
    const message = document.createElement('div');
    message.classList.add('message', sender);
    message.textContent = text;

    chatBox.appendChild(message);
    // Scroll to the latest message
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Voice recognition setup
const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();

recognition.onstart = function() {
    console.log('Voice recognition started. Speak into the microphone.');
};

recognition.onresult = async function(event) {
    const transcript = event.results[0][0].transcript;
    chatInput.value = transcript; // Set the recognized speech to the input field
    await sendMessage(); // Automatically send the message

    // Optional: Send the transcript to the Groq API for additional processing
    // const groqApiResponse = await fetch('YOUR_GROQ_API_URL', {
    //     method: 'POST',
    //     headers: {
    //         'Content-Type': 'application/json',
    //     },
    //     body: JSON.stringify({ transcript: transcript }),
    // });
    // // Handle the response from the Groq API if needed
};

recognition.onerror = function(event) {
    console.error('Error occurred in recognition: ' + event.error);
};

// Start voice recognition when the voice button is clicked
voiceBtn.addEventListener('click', function() {
    recognition.start();
});