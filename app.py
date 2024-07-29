import streamlit as st
import random
from transformers import AutoModelForCausalLM, AutoTokenizer

class AI_Chatbot:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small", padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")

    def get_response(self, user_input):
        try:
            input_ids = self.tokenizer.encode(user_input + self.tokenizer.eos_token, return_tensors='pt')
            print("Input IDs:", input_ids)  # Debugging
            bot_output = self.model.generate(input_ids, max_length=1000, pad_token_id=self.tokenizer.eos_token_id)
            print("Bot Output:", bot_output)  # Debugging
            response = self.tokenizer.decode(bot_output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
            return response
        except Exception as e:
            return f"Error: {str(e)}"

# Create instance of the chatbot
chatbot = AI_Chatbot()

def main():
    st.title("Mayuri Billing Software")

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
    podi_idli = st.sidebar.number_input("Podi Idli (3 pcs) - $9", min_value=0)
    idli_sambar = st.sidebar.number_input("Idli Sambar (3 pcs) - $9", min_value=0)
    pongal = st.sidebar.number_input("Pongal - $10", min_value=0)
    aloo_puri = st.sidebar.number_input("Aloo Puri - $13", min_value=0)

    st.sidebar.header("Chaat and Pakoda")
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

        st.write("### Bill Summary")
        st.write(f"**Bill Number:** {bill_no}")
        st.write(f"**Customer Name:** {c_name}")
        st.write(f"**Phone Number:** {phone}")
        st.write(f"**Total Appetizers Price:** ${total_appetizers_price}")
        st.write(f"**Total Main Dishes Price:** ${total_main_dishes_price}")
        st.write(f"**Total Chaat & Pakoda Price:** ${total_chaat_pakoda_price}")
        st.write(f"**Total Tax:** ${tax}")
        st.write(f"**Service Charge:** ${service_charge}")
        st.write(f"**Total Bill:** ${total_bill}")

    st.sidebar.header("Chat with AI")
    user_input = st.sidebar.text_input("You")
    if st.sidebar.button("Send"):
        if user_input:
            response = chatbot.get_response(user_input)
            st.sidebar.write(f"**Bot:** {response}")

if __name__ == "__main__":
    main()
