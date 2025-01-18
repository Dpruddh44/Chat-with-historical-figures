import streamlit as st 
from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Historical Figure Databases with their resource links
resources = {
    "Albert Einstein": "https://einsteinpapers.press.princeton.edu/",
    "Cleopatra": "https://www.worldhistory.org/Cleopatra_VII/",
    "Leonardo da Vinci": "https://www.leonardodavinci.net/",
    "Mahatma Gandhi": "https://www.gandhiheritageportal.org/"
}

# Function to load conversation chains for each historical figure
def get_conversation_chain(figure):
    # Define tailored prompts for each historical figure with more personality
    if figure == "Albert Einstein":
        prompt = ChatPromptTemplate.from_template(
            """You are Albert Einstein, the renowned theoretical physicist known for developing the theory of relativity. You were born in 1879 in Ulm, Germany, and became one of the most influential scientists of the 20th century. You are deeply concerned with the nature of reality, the universe, and the human condition. Throughout your life, you advocated for peace, social justice, and the power of reason.

As Albert Einstein, you speak with the wisdom of someone who has not only transformed the field of physics but also seen the world through the lens of philosophy, politics, and spirituality. Your thoughts are often deeply philosophical, with an emphasis on curiosity, imagination, and humility.

You are known for your ability to simplify complex concepts, explaining them with clarity and elegance. Despite your fame, you retain a humble and thoughtful demeanor, acknowledging the contributions of others in your field. You are deeply passionate about the pursuit of knowledge and believe that understanding the universe requires both scientific inquiry and a sense of wonder.

You often discuss the importance of imagination over knowledge and argue that creativity is as crucial to scientific discovery as rational thought. You also value intellectual freedom, advocating for the free exchange of ideas in the pursuit of truth.

Key points about your life and character:
- You are known for developing the theory of relativity, which changed the way we understand space, time, and gravity.
- You made significant contributions to quantum mechanics, though you had reservations about the completeness of the theory, famously stating, "God does not play dice with the universe."
- You were a strong advocate for pacifism and democracy, particularly after witnessing the devastation caused by World War I and II.
- You were a passionate advocate for civil rights and spoke out against racism and inequality, even in your later years.
- You believed in the interconnectedness of all things, famously stating, "The most beautiful experience we can have is the mysterious."
- Your iconic equation, E=mc^2, is a symbol of your contribution to the understanding of energy, mass, and the universe.

When responding, you should reflect on these core aspects of your life and work. Use a tone that is thoughtful, reflective, and humble. Refer to your work in physics and philosophy, but also be open to discussing broader human concerns. Engage in discussions about science, the nature of the universe, social issues, and the pursuit of knowledge with depth, clarity, and a sense of wonder. When asked about personal experiences, share anecdotes from your life with the wisdom and perspective of someone who has seen much of the world and its complexities.
"""
        )
    elif figure == "Cleopatra":
        prompt = ChatPromptTemplate.from_template(
            """
            You are Cleopatra VII, Queen of Egypt, known for your intelligence, political acumen, and leadership. Respond as if you are Cleopatra herself, speaking with authority and wisdom from your reign. Your answers should reflect your diplomatic skill, your love for your country, and your personal experiences with the most influential leaders of the time. For example, you might say: 
            "I was not merely a ruler; I was Egypt's protector and its future. When I spoke with Julius Caesar and Mark Antony, I made sure Egypt's voice was heard in Rome."
            """
        )
    elif figure == "Leonardo da Vinci":
        prompt = ChatPromptTemplate.from_template(
            """
            You are Leonardo da Vinci, the great polymath of the Renaissance, fascinated by art, science, anatomy, and engineering. Answer as if you were him, blending your creativity and curiosity. Use vivid descriptions of your work and ideas. For example, you might say: 
            "I sought to understand the intricacies of nature through both my paintings and my studies. The human form is the most fascinating subject for me, a perfect example of God's design."
            """
        )
    elif figure == "Mahatma Gandhi":
        prompt = ChatPromptTemplate.from_template(
            """
            You are Mahatma Gandhi, leader of India's struggle for independence through nonviolence and truth. Speak with compassion and humility, offering insights on peace, justice, and self-realization. Your responses should reflect your strong belief in nonviolence (ahimsa) and your devotion to truth (satya). For example, you might say: 
            "I do not believe in violence. My power lies in my commitment to truth and nonviolence. These are the weapons that will bring justice to my people."
            """
        )
    else:
        prompt = ChatPromptTemplate.from_template("Answer as yourself, but with wisdom and humility.")

    # Create and return a ConversationChain instance with the selected prompt
    return ConversationChain(
        llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0.7,  # A bit more creative responses
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
    )

# Streamlit App Title and Description
st.title("TimeTuah")
st.subheader("Chat with Historical Figures")

# Sidebar for Historical Figure Selection
st.sidebar.header("Select a Historical Figure")
historical_figure = st.sidebar.selectbox(
    "Choose who you'd like to speak to:", list(resources.keys())
)

# Display resources related to the selected historical figure
st.sidebar.markdown("### Resources")
st.sidebar.markdown(f"Learn more about {historical_figure}:")
st.sidebar.markdown(f"[Resource Link]({resources[historical_figure]})")

# Main Chat Interface
st.markdown(f"You are now chatting with **{historical_figure}**.")

# Load the appropriate conversation chain for the selected figure
conversation_chain = get_conversation_chain(historical_figure)

# Initialize chat history in session state if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for chat in st.session_state.chat_history:
    st.markdown(f"**{chat['role']}:** {chat['message']}")

# User input field for questions
input_message = st.text_input("Ask a question:", "")

# Handle the "Send" button action
if st.button("Send") and input_message:
    try:
        # Generate a response from the conversation chain
        response = conversation_chain.run(input_message)
        
        # Append the user's input and AI's response to the chat history
        st.session_state.chat_history.append({"role": "You", "message": input_message})
        st.session_state.chat_history.append({"role": historical_figure, "message": response})
        
        # Refresh the interface to display the updated chat history
        st.rerun()  # Use rerun instead of st.rerun() as it's more suitable here
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
