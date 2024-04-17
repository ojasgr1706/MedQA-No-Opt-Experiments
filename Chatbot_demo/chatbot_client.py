
def send_to_server(user_input,algo):
    # send instring and algorithm to server
    # await and return response
    
def create_options():
    while(output['not_enough_options']):
        # send instring for option to client
        # await output from server
        # print option


if 'submit' not in st.session_state:
    st.session_state.submit = False
if 'submit_backward' not in st.session_state:
    st.session_state.submit_backward = False

st.title("Choose the algorithm you want to run :")
algorithms = ["MedCodex - Greedy", "Codex - Greedy", "MedCodex + Codex (F+B)", "MedCodex + Verifier (F + RM)"]
# algo = algorithms[option_ind]
# Drop-down menu with four options
algo = st.selectbox("Choose an option:", ["MedCodex - Greedy", "Codex - Greedy", "MedCodex + Codex (F+B)", "MedCodex + Verifier (F + RM)"])

# SEND algo TO SERVER

# Display the selected option
# print("You selected:", algo)
st.write("You selected:", algo)

user_input = st.text_area("Enter input prompt", value="", height=50, max_chars=None)
# user_input = input("User input here : ")
# print("The input is :\n",user_input)
# st.write("The input is :\n",user_input)

submit = st.button("Submit input")

if submit:
    st.session_state.submit = True
    
if algo == "MedCodex - Greedy" or algo == "Codex - Greedy":
    if algo == "MedCodex - Greedy":
        send_to_server(user_input, 'medcodex')

    elif algo == "Codex - Greedy":
        send_to_server(user_input, 'codex')
        
    output = await from server
    
elif algo == "MedCodex + Codex (F+B)" or algo == "MedCodex + Verifier (F + RM)":
    
    options = await create_options()
    
    if (len(uniq_options) < 4):
            # output = "Not enough options generated!"
            st.write("Not enough Options")
            # print("Not enough Options")

        else:
            if algo == "MedCodex + Codex (F+B)":
                
