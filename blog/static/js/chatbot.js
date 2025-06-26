document.addEventListener("DOMContentLoaded", function() {
    const chatInput = document.querySelector(".chat-input textarea");
    const sendChatBtn = document.querySelector("#send-btn");
    const chatForm = document.querySelector("#chat-form");
    const chatbox = document.querySelector(".chatbox");

    const createChatLi = (message, className) => {
        const chatLi = document.createElement("li");
        chatLi.classList.add("chat", className);
        let chatContent = className === "outgoing" 
            ? `<p></p>` 
            : `<span class="material-symbols-outlined"></span><p></p>`;
        chatLi.innerHTML = chatContent;
        chatLi.querySelector("p").textContent = message;
        return chatLi;
    }

    const handleChat = async (event) => {
        event.preventDefault(); // Prevent default form submission

        const formData = new FormData(chatForm);
        const response = await fetch(chatForm.action, {
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': document.querySelector('[name=csrfmiddlewaretoken]').value
            }
        });

        const data = await response.json();
        const userMessage = chatInput.value.trim();
        if (!userMessage) return;

        chatInput.value = "";
        chatbox.appendChild(createChatLi(userMessage, "outgoing"));
        chatbox.scrollTo(0, chatbox.scrollHeight);

        const incomingChatLi = createChatLi(data.response, "incoming");
        chatbox.appendChild(incomingChatLi);
        chatbox.scrollTo(0, chatbox.scrollHeight);
    }
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
          e.preventDefault();
          handleChat(e);
      }
  });
    sendChatBtn.addEventListener("click", handleChat);
    chatForm.addEventListener("submit", handleChat);
});