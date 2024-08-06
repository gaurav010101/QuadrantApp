import streamlit as st
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class AI_Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        self.responses = {
            "best dish": "The top-selling dish at our restaurant is the Masala Dosa. It's a favorite among our customers!",
            "menu": "Here are some of our menu items:\n- Appetizers: Chilli Baby Corn, Chilli Paneer, Chilli Gobi, etc.\n- Main Dishes: Plain Dosa, Masala Dosa, Pongal, etc.\n- Chaat & Pakoda: Samosa Chaat, Pani Puri, Sev Puri, etc.",
            "opening hours": "We are open from 10 AM to 10 PM every day.",
            "location": "We are located at 123 Food Street, Flavor Town.",
            "reservation": "You can call us at (123) 456-7890 to make a reservation.",
            "specials": "Today's specials include the spicy Paneer Tikka and the refreshing Mango Lassi.",
            "allergies": "Please let us know about any allergies, and we will ensure that your meal is safe for you.",
            "price": "Here are some prices:\n- Chilli Baby Corn: $12\n- Plain Dosa: $10\n- Masala Dosa: $13\n- Samosa Chaat: $9",
            "contact": "You can contact us at (123) 456-7890 or email us at info@restaurant.com.",
            "thank you": "You're welcome! If you have any other questions, feel free to ask."
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
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
            bot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
            response = self.tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# Create instance of the chatbot
chatbot = AI_Chatbot()

def main():
    # Apply custom CSS to style the app
    st.markdown(
        """
        <style>
        .title-box {
            background: linear-gradient(135deg, #ff7e5f, #feb47b); /* Gradient background */
            color: #fff;
            padding: 20px;
            border-radius: 20px;
            text-align: center;
            font-size: 32px;
            margin-bottom: 40px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.5);
            animation: bounce 2s infinite;
        }
        .billing-box {
            background: radial-gradient(circle, #fff3e6, #ffcccb); /* Radial gradient */
            border: 5px double #ff5733; /* Double border */
            border-radius: 20px;
            padding: 30px;
            margin: 40px 0;
            box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.3);
            animation: slideIn 1.5s ease-out;
        }
        .billing-summary {
            font-size: 24px;
            font-weight: bold;
            color: #ff5733;
            margin-bottom: 20px;
        }
        .billing-item {
            font-size: 20px;
            margin: 10px 0;
        }
        .sidebar-content {
            background: linear-gradient(135deg, #f9d423, #ff4e50); /* Gradient background */
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 8px 16px rgba(0, 0, 0, 0.2);
        }
        .chatbox {
            background: linear-gradient(135deg, #ff6f61, #d57e7a); /* Gradient background */
            border: 4px dashed #ffcc00; /* Dashed border */
            border-radius: 25px;
            padding: 30px;
            margin-top: 25px;
            box-shadow: 0px 12px 24px rgba(0, 0, 0, 0.4);
            animation: spin 3s linear infinite;
        }
        .user-input {
            border: 3px solid #ffcc00; /* Bright yellow border */
            border-radius: 15px;
            padding: 15px;
            font-size: 18px;
            margin-bottom: 15px;
            background-color: #fff;
            transition: all 0.3s ease;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .user-input:focus {
            border-color: #ff6f61; /* Focus border color */
            box-shadow: 0 0 12px #ff6f61; /* Glow effect on focus */
        }
        .bot-response {
            color: #ff6f61; /* Bright coral color */
            font-size: 20px;
            font-weight: bold;
            background: linear-gradient(135deg, #fff, #f0f0f0); /* Gradient background */
            border-left: 6px solid #ff6f61; /* Highlight border on the left */
            padding: 12px;
            border-radius: 15px;
            margin-top: 15px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        }
        @keyframes bounce {
            0%, 20%, 50%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-30px); }
            60% { transform: translateY(-15px); }
        }
        @keyframes slideIn {
            from { transform: translateX(-100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        </style>
        <div class="title-box">Mayuri Restaurant</div>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.sidebar.header("Customer Details")
    c_name = st.sidebar.text_input("Customer Name")
    phone = st.sidebar.text_input("Phone Number")
    bill_no = st.sidebar.text_input("Order Number", value=str(random.randint(1000, 9999)))

    st.sidebar.header("Appetizers")
    chilli_baby_corn = st.sidebar.number_input("Chilli Baby Corn - $12", min_value=0)
    chilli_paneer = st.sidebar.number_input("Chilli Paneer - $14", min_value=0)
    chill_gobi = st.sidebar.number_input("Chilli Gobi - $13", min_value=0)
    gobi_manchurian = st.sidebar.number_input("Gobi Manchurian - $12", min_value=0)
    paneer_manchurian = st.sidebar.number_input("Paneer Manchurian - $14", min_value=0)
    paneer_65 = st.sidebar.number_input("Paneer 65 - $14", min_value=0)
    paneer_kathi_roll = st.sidebar.number_input("Paneer Kathi Roll - $10", min_value=0)

    st.sidebar.header("Main Dishes")
    plain_ghee_dosa = st.sidebar.number_input("Plain Dosa/Ghee Dosa - $10", min_value=0)
    podi_dosa = st.sidebar.number_input("Podi Dosa - $13", min_value=0)
    masala_dosa = st.sidebar.number_input("Masala Dosa - $13", min_value=0)
    podi_idli = st.sidebar.number_input("Podi Idli - $9", min_value=0)
    idli_sambar = st.sidebar.number_input("Idli Sambar - $9", min_value=0)
    pongal = st.sidebar.number_input("Pongal - $10", min_value=0)
    aloo_puri = st.sidebar.number_input("Aloo Puri - $13", min_value=0)

    st.sidebar.header("Chaat & Pakoda")
    samosa_chaat = st.sidebar.number_input("Samosa Chaat - $9", min_value=0)
    papri_chaat = st.sidebar.number_input("Papri Chaat - $9", min_value=0)
    vada_pav = st.sidebar.number_input("Vada Pav - $8", min_value=0)
    pani_puri = st.sidebar.number_input("Pani Puri - $8", min_value=0)
    sev_puri = st.sidebar.number_input("Sev Puri - $9", min_value=0)
    aloo_tiki = st.sidebar.number_input("Aloo Tiki - $9", min_value=0)
    mirchi_chaat = st.sidebar.number_input("Mirchi Chaat - $9", min_value=0)

    if st.sidebar.button("Generate Bill"):
        total_appetizers_price = (
            chilli_baby_corn * 12 +
            chilli_paneer * 14 +
            chill_gobi * 13 +
            gobi_manchurian * 12 +
            paneer_manchurian * 14 +
            paneer_65 * 14 +
            paneer_kathi_roll * 10
        )
        total_main_dishes_price = (
            plain_ghee_dosa * 10 +
            podi_dosa * 13 +
            masala_dosa * 13 +
            podi_idli * 9 +
            idli_sambar * 9 +
            pongal * 10 +
            aloo_puri * 13
        )
        total_chaat_pakoda_price = (
            samosa_chaat * 9 +
            papri_chaat * 9 +
            vada_pav * 8 +
            pani_puri * 8 +
            sev_puri * 9 +
            aloo_tiki * 9 +
            mirchi_chaat * 9
        )

        tax = (total_appetizers_price + total_main_dishes_price + total_chaat_pakoda_price) * 0.1
        service_charge = 5
        total_bill = total_appetizers_price + total_main_dishes_price + total_chaat_pakoda_price + tax + service_charge

        st.markdown(
            f"""
            <div class="billing-box">
                <div class="billing-summary">Bill Summary</div>
                <div class="billing-item"><strong>Bill Number:</strong> {bill_no}</div>
                <div class="billing-item"><strong>Customer Name:</strong> {c_name}</div>
                <div class="billing-item"><strong>Phone Number:</strong> {phone}</div>
                <div class="billing-item"><strong>Total Appetizers Price:</strong> ${total_appetizers_price}</div>
                <div class="billing-item"><strong>Total Main Dishes Price:</strong> ${total_main_dishes_price}</div>
                <div class="billing-item"><strong>Total Chaat & Pakoda Price:</strong> ${total_chaat_pakoda_price}</div>
                <div class="billing-item"><strong>Total Tax:</strong> ${tax}</div>
                <div class="billing-item"><strong>Service Charge:</strong> ${service_charge}</div>
                <div class="billing-item"><strong>Total Bill:</strong> ${total_bill}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.sidebar.markdown('<div class="chatbox">', unsafe_allow_html=True)

    st.sidebar.header("Chat with AI")
    user_input = st.sidebar.text_input("You", key="chat_input", placeholder="Type your message here...", help="Ask me anything!", className="user-input")
    if st.sidebar.button("Send"):
        if user_input:
            response = chatbot.get_response(user_input)
            st.sidebar.markdown(f"<div class='bot-response'><strong>Bot:</strong> {response}</div>", unsafe_allow_html=True)

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
