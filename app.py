import streamlit as st
import random
from transformers import BlenderbotSmallTokenizer, BlenderbotForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

class AI_Chatbot:
    def __init__(self):
        self.tokenizer = BlenderbotSmallTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
        self.model = BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")
        self.responses = {
            "best dish": "The top-selling dish at our restaurant is the Masala Dosa. It's a favorite among our customers!",
            "menu": "Here are some of our menu items:\n- Appetizers: Chilli Baby Corn, Chilli Paneer, Chilli Gobi, etc.\n- Main Dishes: Plain Dosa, Masala Dosa, Pongal, etc.\n- Chaat & Pakoda: Samosa Chaat, Pani Puri, Sev Puri, etc.",
            "opening hours": "We are open from 10 AM to 10 PM every day.",
            "location": "We are located at 123 Food Street, Flavor Town.",
            "reservation": "You can call us at (123) 456-7890 to make a reservation.",
            "special": "Today's specials include the spicy Paneer 65 and chat masala.",
            "allergies": "Please let us know about any allergies, and we will ensure that your meal is safe for you.",
            "price": "Here are some prices:\n- Chilli Baby Corn: $12\n- Plain Dosa: $10\n- Masala Dosa: $13\n- Samosa Chaat: $9",
            "contact": "You can contact us at (123) 456-7890 or email us at info@restaurant.com.",
            "thank you": "You're welcome! If you have any other questions, feel free to ask.",
            "what goes well with that": "The Gobi Manchurian goes great with the Masala Dosa! It is even recommended by this store's owner."
        }
        self.vectorizer = TfidfVectorizer().fit(self.responses.keys())

    def get_response(self, user_input):
        user_input_lower = user_input.lower()
        
        # Check for predefined responses first
        for key, response in self.responses.items():
            if key in user_input_lower:
                return response
        
        # If no predefined response is found, calculate the similarity to predefined keys
        input_vector = self.vectorizer.transform([user_input_lower])
        cosine_similarities = cosine_similarity(input_vector, self.vectorizer.transform(self.responses.keys())).flatten()
        highest_similarity_idx = cosine_similarities.argmax()
        
        if cosine_similarities[highest_similarity_idx] > 0.1:
            closest_key = list(self.responses.keys())[highest_similarity_idx]
            return self.responses[closest_key]

        # Fallback to model-based response for general queries
        try:
            input_ids = self.tokenizer.encode(user_input, return_tensors='pt')
            bot_output = self.model.generate(input_ids, max_length=100)
            response = self.tokenizer.decode(bot_output[0], skip_special_tokens=True)
            
            # Prevent repetitive responses
            if response.strip() == "":
                return "Sorry, I didn't understand that. Can you please ask something else?"
            return response
        except Exception as e:
            return f"Error: {str(e)}"

def create_db():
    conn = sqlite3.connect('billing2.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS bills (
                        order_number TEXT PRIMARY KEY,
                        customer_name TEXT,
                        phone_number TEXT,
                        chilli_baby_corn INTEGER,
                        chilli_paneer INTEGER,
                        chill_gobi INTEGER,
                        gobi_manchurian INTEGER,
                        paneer_manchurian INTEGER,
                        paneer_65 INTEGER,
                        paneer_kathi_roll INTEGER,
                        plain_ghee_dosa INTEGER,
                        podi_dosa INTEGER,
                        masala_dosa INTEGER,
                        podi_idli INTEGER,
                        idli_sambar INTEGER,
                        pongal INTEGER,
                        aloo_puri INTEGER,
                        samosa_chaat INTEGER,
                        papri_chaat INTEGER,
                        vada_pav INTEGER,
                        pani_puri INTEGER,
                        sev_puri INTEGER,
                        aloo_tiki INTEGER,
                        mirchi_chaat INTEGER,
                        appetizers_total TEXT,
                        main_dishes_total TEXT,
                        chaat_pakoda_total TEXT,
                        appetizers_total_tax TEXT,
                        main_dishes_total_tax TEXT,
                        chaat_pakoda_total_tax TEXT,
                        total_all_bil TEXT
                    )''')
    conn.commit()
    conn.close()

def main():
    # Apply custom CSS to style the app
    st.markdown(
        """
        <style>
        .title-box {
            background-color: #ff5733; /* Bright red-orange */
            color: white;
            padding: 15px;
            border-radius: 10px;
            text-align: center;
            font-size: 28px;
            margin-bottom: 30px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.3);
            animation: pulsate 2s infinite;
        }
        .billing-box {
            background-color: #fff3e6; /* Soft peach */
            border: 3px solid #ff5733; /* Bright red-orange */
            border-radius: 15px;
            padding: 25px;
            margin: 30px 0;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        }
        .billing-summary {
            font-size: 20px;
            font-weight: bold;
            color: #ff5733; /* Bright red-orange */
            margin-bottom: 15px;
        }
        .billing-item {
            font-size: 18px;
            margin: 8px 0;
        }
        .sidebar-content {
            background-color: #f7f7f7;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .chatbox {
            background: linear-gradient(135deg, #ffcc00, #ff3399); /* Gradient background */
            border: 4px dashed #1e90ff; /* Dashed border */
            border-radius: 20px;
            padding: 20px;
            margin-top: 20px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.3);
            animation: fadeIn 2s ease-in-out;
        }
        .user-input {
            border: 2px solid #1e90ff; /* Dodger blue */
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
            margin-bottom: 10px;
            background-color: #fff;
            transition: all 0.3s ease;
        }
        .user-input:focus {
            border-color: #ff3399; /* Highlight border on focus */
            box-shadow: 0 0 10px #ff3399; /* Glow effect on focus */
        }
        .bot-response {
            color: #1e90ff; /* Dodger blue */
            font-size: 18px;
            font-weight: bold;
            background-color: #e6f7ff; /* Light blue background */
            border-left: 4px solid #1e90ff; /* Highlight border on the left */
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
        }
        .section-title {
            background-color: #ffcc00; /* Bright yellow */
            color: #fff;
            padding: 10px;
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
        }
        .input-item {
            background-color: #fff;
            border: 2px solid #ff5733; /* Bright red-orange */
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        @keyframes pulsate {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        </style>
        <div class="title-box">Mayuri Restaurant</div>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for customer details and chat
    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.sidebar.header("Customer Details")
    c_name = st.sidebar.text_input("Customer Name", key="c_name", placeholder="Enter your name", help="Enter the customer's name")
    phone = st.sidebar.text_input("Phone Number", key="phone", placeholder="Enter phone number", help="Enter the customer's phone number")
    
    st.sidebar.header("Chat with Our Bot")
    st.sidebar.markdown('<div class="chatbox">', unsafe_allow_html=True)
    
    user_input = st.sidebar.text_input("Ask a question", key="user_input", placeholder="Type your question here", help="Type a question to ask our chatbot")
    chatbot = AI_Chatbot()
    
    if user_input:
        bot_response = chatbot.get_response(user_input)
        st.sidebar.markdown(f'<div class="bot-response">{bot_response}</div>', unsafe_allow_html=True)
    
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="billing-box">', unsafe_allow_html=True)
    st.header("Mayuri Restaurant Billing System")
    
    # Billing section
    appetizers = {
        "Chilli Baby Corn": st.number_input("Chilli Baby Corn", min_value=0, max_value=10, key="chilli_baby_corn", step=1),
        "Chilli Paneer": st.number_input("Chilli Paneer", min_value=0, max_value=10, key="chilli_paneer", step=1),
        "Chilli Gobi": st.number_input("Chilli Gobi", min_value=0, max_value=10, key="chill_gobi", step=1),
        "Gobi Manchurian": st.number_input("Gobi Manchurian", min_value=0, max_value=10, key="gobi_manchurian", step=1),
        "Paneer Manchurian": st.number_input("Paneer Manchurian", min_value=0, max_value=10, key="paneer_manchurian", step=1),
        "Paneer 65": st.number_input("Paneer 65", min_value=0, max_value=10, key="paneer_65", step=1),
        "Paneer Kathi Roll": st.number_input("Paneer Kathi Roll", min_value=0, max_value=10, key="paneer_kathi_roll", step=1),
    }
    
    main_dishes = {
        "Plain Ghee Dosa": st.number_input("Plain Ghee Dosa", min_value=0, max_value=10, key="plain_ghee_dosa", step=1),
        "Podi Dosa": st.number_input("Podi Dosa", min_value=0, max_value=10, key="podi_dosa", step=1),
        "Masala Dosa": st.number_input("Masala Dosa", min_value=0, max_value=10, key="masala_dosa", step=1),
        "Podi Idli": st.number_input("Podi Idli", min_value=0, max_value=10, key="podi_idli", step=1),
        "Idli Sambar": st.number_input("Idli Sambar", min_value=0, max_value=10, key="idli_sambar", step=1),
        "Pongal": st.number_input("Pongal", min_value=0, max_value=10, key="pongal", step=1),
        "Aloo Puri": st.number_input("Aloo Puri", min_value=0, max_value=10, key="aloo_puri", step=1),
    }
    
    chaat_pakoda = {
        "Samosa Chaat": st.number_input("Samosa Chaat", min_value=0, max_value=10, key="samosa_chaat", step=1),
        "Papri Chaat": st.number_input("Papri Chaat", min_value=0, max_value=10, key="papri_chaat", step=1),
        "Vada Pav": st.number_input("Vada Pav", min_value=0, max_value=10, key="vada_pav", step=1),
        "Pani Puri": st.number_input("Pani Puri", min_value=0, max_value=10, key="pani_puri", step=1),
        "Sev Puri": st.number_input("Sev Puri", min_value=0, max_value=10, key="sev_puri", step=1),
        "Aloo Tiki": st.number_input("Aloo Tiki", min_value=0, max_value=10, key="aloo_tiki", step=1),
        "Mirchi Chaat": st.number_input("Mirchi Chaat", min_value=0, max_value=10, key="mirchi_chaat", step=1),
    }
    
    st.markdown('<div class="billing-summary">Summary:</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="billing-item">Appetizers Total: ${}</div>'.format(sum(appetizers.values()) * 10), unsafe_allow_html=True)
    st.markdown('<div class="billing-item">Main Dishes Total: ${}</div>'.format(sum(main_dishes.values()) * 10), unsafe_allow_html=True)
    st.markdown('<div class="billing-item">Chaat & Pakoda Total: ${}</div>'.format(sum(chaat_pakoda.values()) * 10), unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    create_db()
    main()
