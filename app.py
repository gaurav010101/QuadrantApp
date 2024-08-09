import streamlit as st
import random
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sqlite3

class AI_Chatbot:
    def __init__(self):
        self.tokenizer = BlenderbotTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
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
            bot_output = self.model.generate(input_ids, max_length=100, num_return_sequences=1)
            response = self.tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True).strip()

            if not response:
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
        <div class="title-box">Restaurant Billing System</div>
        """,
        unsafe_allow_html=True
    )

    st.title("Restaurant Billing System")

    # Create and initialize the database
    create_db()

    # Sidebar for chatbot
    st.sidebar.header("Chat with Our Bot")
    chatbot = AI_Chatbot()
    user_input = st.sidebar.text_input("Type your question here:")
    if user_input:
        response = chatbot.get_response(user_input)
        st.sidebar.markdown(f"<div class='bot-response'>{response}</div>", unsafe_allow_html=True)

    # Right side of the app for menu
    st.sidebar.header("Customer Information")
    customer_name = st.sidebar.text_input("Customer Name")
    phone_number = st.sidebar.text_input("Phone Number")

    # Billing details
    st.sidebar.header("Billing Details")
    chilli_baby_corn_qty = st.sidebar.number_input("Chilli Baby Corn", min_value=0, value=0)
    chilli_paneer_qty = st.sidebar.number_input("Chilli Paneer", min_value=0, value=0)
    chill_gobi_qty = st.sidebar.number_input("Chilli Gobi", min_value=0, value=0)
    gobi_manchurian_qty = st.sidebar.number_input("Gobi Manchurian", min_value=0, value=0)
    paneer_manchurian_qty = st.sidebar.number_input("Paneer Manchurian", min_value=0, value=0)
    paneer_65_qty = st.sidebar.number_input("Paneer 65", min_value=0, value=0)
    paneer_kathi_roll_qty = st.sidebar.number_input("Paneer Kathi Roll", min_value=0, value=0)
    plain_ghee_dosa_qty = st.sidebar.number_input("Plain Ghee Dosa", min_value=0, value=0)
    podi_dosa_qty = st.sidebar.number_input("Podi Dosa", min_value=0, value=0)
    masala_dosa_qty = st.sidebar.number_input("Masala Dosa", min_value=0, value=0)
    podi_idli_qty = st.sidebar.number_input("Podi Idli", min_value=0, value=0)
    idli_sambar_qty = st.sidebar.number_input("Idli Sambar", min_value=0, value=0)
    pongal_qty = st.sidebar.number_input("Pongal", min_value=0, value=0)
    aloo_puri_qty = st.sidebar.number_input("Aloo Puri", min_value=0, value=0)
    samosa_chaat_qty = st.sidebar.number_input("Samosa Chaat", min_value=0, value=0)
    papri_chaat_qty = st.sidebar.number_input("Papri Chaat", min_value=0, value=0)
    vada_pav_qty = st.sidebar.number_input("Vada Pav", min_value=0, value=0)
    pani_puri_qty = st.sidebar.number_input("Pani Puri", min_value=0, value=0)
    sev_puri_qty = st.sidebar.number_input("Sev Puri", min_value=0, value=0)
    aloo_tiki_qty = st.sidebar.number_input("Aloo Tiki", min_value=0, value=0)
    mirchi_chaat_qty = st.sidebar.number_input("Mirchi Chaat", min_value=0, value=0)

    if st.sidebar.button("Generate Bill"):
        # Calculate totals and save to database
        appetizers_total = (chilli_baby_corn_qty * 12) + (chilli_paneer_qty * 15) + (chill_gobi_qty * 12) + (gobi_manchurian_qty * 14) + (paneer_manchurian_qty * 15) + (paneer_65_qty * 16) + (paneer_kathi_roll_qty * 18)
        main_dishes_total = (plain_ghee_dosa_qty * 10) + (podi_dosa_qty * 11) + (masala_dosa_qty * 13) + (podi_idli_qty * 9) + (idli_sambar_qty * 10) + (pongal_qty * 12) + (aloo_puri_qty * 11)
        chaat_pakoda_total = (samosa_chaat_qty * 9) + (papri_chaat_qty * 8) + (vada_pav_qty * 7) + (pani_puri_qty * 6) + (sev_puri_qty * 7) + (aloo_tiki_qty * 8) + (mirchi_chaat_qty * 9)
        total_all_bill = appetizers_total + main_dishes_total + chaat_pakoda_total

        # Create a bill record
        conn = sqlite3.connect('billing2.db')
        cursor = conn.cursor()
        cursor.execute('''INSERT INTO bills (
                            order_number, customer_name, phone_number,
                            chilli_baby_corn, chilli_paneer, chill_gobi,
                            gobi_manchurian, paneer_manchurian, paneer_65,
                            paneer_kathi_roll, plain_ghee_dosa, podi_dosa,
                            masala_dosa, podi_idli, idli_sambar, pongal,
                            aloo_puri, samosa_chaat, papri_chaat, vada_pav,
                            pani_puri, sev_puri, aloo_tiki, mirchi_chaat,
                            appetizers_total, main_dishes_total, chaat_pakoda_total,
                            total_all_bill
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (str(random.randint(1000, 9999)), customer_name, phone_number,
                         chilli_baby_corn_qty, chilli_paneer_qty, chill_gobi_qty,
                         gobi_manchurian_qty, paneer_manchurian_qty, paneer_65_qty,
                         paneer_kathi_roll_qty, plain_ghee_dosa_qty, podi_dosa_qty,
                         masala_dosa_qty, podi_idli_qty, idli_sambar_qty, pongal_qty,
                         aloo_puri_qty, samosa_chaat_qty, papri_chaat_qty, vada_pav_qty,
                         pani_puri_qty, sev_puri_qty, aloo_tiki_qty, mirchi_chaat_qty,
                         appetizers_total, main_dishes_total, chaat_pakoda_total,
                         total_all_bill))
        conn.commit()
        conn.close()

        # Display the bill
        st.markdown(
            f"""
            <div class="billing-box">
            <div class="billing-summary">Bill Summary:</div>
            <div class="billing-item">Appetizers Total: ${appetizers_total}</div>
            <div class="billing-item">Main Dishes Total: ${main_dishes_total}</div>
            <div class="billing-item">Chaat & Pakoda Total: ${chaat_pakoda_total}</div>
            <div class="billing-item">Total Bill: ${total_all_bill}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Display the menu on the right side of the app
    st.markdown(
        """
        <div class="section-title">Menu</div>
        <div class="input-item">
            <h4>Appetizers</h4>
            <p>Chilli Baby Corn - $12</p>
            <p>Chilli Paneer - $15</p>
            <p>Chilli Gobi - $12</p>
            <p>Gobi Manchurian - $14</p>
            <p>Paneer Manchurian - $15</p>
            <p>Paneer 65 - $16</p>
            <p>Paneer Kathi Roll - $18</p>
        </div>
        <div class="input-item">
            <h4>Main Dishes</h4>
            <p>Plain Ghee Dosa - $10</p>
            <p>Podi Dosa - $11</p>
            <p>Masala Dosa - $13</p>
            <p>Podi Idli - $9</p>
            <p>Idli Sambar - $10</p>
            <p>Pongal - $12</p>
            <p>Aloo Puri - $11</p>
        </div>
        <div class="input-item">
            <h4>Chaat & Pakoda</h4>
            <p>Samosa Chaat - $9</p>
            <p>Papri Chaat - $8</p>
            <p>Vada Pav - $7</p>
            <p>Pani Puri - $6</p>
            <p>Sev Puri - $7</p>
            <p>Aloo Tiki - $8</p>
            <p>Mirchi Chaat - $9</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
