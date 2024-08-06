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
            "special": "Today's specials include the spicy Paneer Tikka and the refreshing Mango Lassi.",
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
            
            # Prevent repetitive responses
            if response.strip() == "":
                return "Sorry, I didn't understand that. Can you please ask something else?"
            return response
        except Exception as e:
            return f"Error: {str(e)}"

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
    phone = st.sidebar.text_input("Phone Number", key="phone", placeholder="Enter phone number", help="Enter the phone number")
    bill_no = st.sidebar.text_input("Order Number", value=str(random.randint(1000, 9999)), key="bill_no", placeholder="Enter order number", help="Enter the order number")

    st.sidebar.markdown('<div class="section-title">Appetizers</div>', unsafe_allow_html=True)
    chilli_baby_corn = st.sidebar.number_input("Chilli Baby Corn - $12", min_value=0, key="chilli_baby_corn", help="Enter quantity")
    chilli_paneer = st.sidebar.number_input("Chilli Paneer - $14", min_value=0, key="chilli_paneer", help="Enter quantity")
    chill_gobi = st.sidebar.number_input("Chilli Gobi - $13", min_value=0, key="chill_gobi", help="Enter quantity")
    gobi_manchurian = st.sidebar.number_input("Gobi Manchurian - $12", min_value=0, key="gobi_manchurian", help="Enter quantity")
    paneer_manchurian = st.sidebar.number_input("Paneer Manchurian - $14", min_value=0, key="paneer_manchurian", help="Enter quantity")
    paneer_65 = st.sidebar.number_input("Paneer 65 - $14", min_value=0, key="paneer_65", help="Enter quantity")
    paneer_kathi_roll = st.sidebar.number_input("Paneer Kathi Roll - $10", min_value=0, key="paneer_kathi_roll", help="Enter quantity")

    st.sidebar.markdown('<div class="section-title">Main Dishes</div>', unsafe_allow_html=True)
    plain_ghee_dosa = st.sidebar.number_input("Plain Dosa/Ghee Dosa - $10", min_value=0, key="plain_ghee_dosa", help="Enter quantity")
    podi_dosa = st.sidebar.number_input("Podi Dosa - $13", min_value=0, key="podi_dosa", help="Enter quantity")
    masala_dosa = st.sidebar.number_input("Masala Dosa - $13", min_value=0, key="masala_dosa", help="Enter quantity")
    podi_idli = st.sidebar.number_input("Podi Idli (3 pcs) - $9", min_value=0, key="podi_idli", help="Enter quantity")
    idli_sambar = st.sidebar.number_input("Idli Sambar (3 pcs) - $9", min_value=0, key="idli_sambar", help="Enter quantity")
    pongal = st.sidebar.number_input("Pongal - $10", min_value=0, key="pongal", help="Enter quantity")
    aloo_puri = st.sidebar.number_input("Aloo Puri - $13", min_value=0, key="aloo_puri", help="Enter quantity")

    st.sidebar.markdown('<div class="section-title">Chaat and Pakoda</div>', unsafe_allow_html=True)
    samosa_chaat = st.sidebar.number_input("Samosa Chaat - $9", min_value=0, key="samosa_chaat", help="Enter quantity")
    papri_chaat = st.sidebar.number_input("Papri Chaat - $9", min_value=0, key="papri_chaat", help="Enter quantity")
    vada_pav = st.sidebar.number_input("Vada Pav - $8", min_value=0, key="vada_pav", help="Enter quantity")
    pani_puri = st.sidebar.number_input("Pani Puri - $8", min_value=0, key="pani_puri", help="Enter quantity")
    sev_puri = st.sidebar.number_input("Sev Puri - $9", min_value=0, key="sev_puri", help="Enter quantity")
    aloo_tiki = st.sidebar.number_input("Aloo Tiki - $9", min_value=0, key="aloo_tiki", help="Enter quantity")
    mirchi_chaat = st.sidebar.number_input("Mirchi Chaat - $9", min_value=0, key="mirchi_chaat", help="Enter quantity")

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    st.sidebar.markdown('<div class="chatbox">', unsafe_allow_html=True)
    
    st.sidebar.header("Chat with AI")
    
    user_input = st.sidebar.text_input("Type your message here...", key="user_input", placeholder="Ask something...", help="Type your query here")

    if st.sidebar.button("Send"):
        response = chatbot.get_response(user_input)
        st.sidebar.markdown(f'<div class="bot-response">{response}</div>', unsafe_allow_html=True)

    st.sidebar.markdown('</div>', unsafe_allow_html=True)

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

        # Display Bill Summary below the Mayuri Restaurant Bar
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

if __name__ == "__main__":
    chatbot = AI_Chatbot()
    main()
