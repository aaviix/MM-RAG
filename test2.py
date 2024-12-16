import streamlit as st

# Set the page configuration with a white background
st.set_page_config(
    page_title="Ad Creator",
    page_icon=":bulb:",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Apply custom CSS to set the background color to white
st.markdown(
    """
    <style>
    .main {
        background-color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# App title
st.title("Advertisement Text Creator")

# User input for text
input_text = st.text_area("Enter your product or service description:")

# Button to create advertisement
if st.button("Create Advertisement"):
    if input_text:
        # Simple transformation logic to create an advertisement
        ad_text = f"🚀 Don't miss out on the best {input_text}! Get yours now! 🌟"
        st.subheader("Your Advertisement:")
        st.write(ad_text)
    else:
        st.warning("Please enter a description to create an advertisement.")

# Display the advertisement text
if 'ad_text' in locals():
    st.write(ad_text)

