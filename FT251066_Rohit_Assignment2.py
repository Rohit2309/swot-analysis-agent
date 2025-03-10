#!/usr/bin/env python
# coding: utf-8

# ### AI-Agents for SWOT Analysis Assignment ###

# ### Rohit Behara - FT251066 ###

# In[1]:


# Installing google-generativeai library used to interact with Google's Gen AI model
#!pip install google-generativeai


# In[2]:


# Installing langchain-google-genai library to integrate Google's Gen AI model with LangChain
#!pip install langchain-google-genai


# In[3]:


# Installing required library to extract text from pdf files
#!pip install PyPDF2


# In[4]:


# importing necessary libraries

import os  # To access and manage environment variables
import google.generativeai as genai  # Google Generative AI library for interacting with Gemini API
import streamlit as st  # Streamlit for building the web user interface
import langchain  # To print langchain version
from langchain_google_genai import ChatGoogleGenerativeAI  # Chat interface for Gemini via LangChain
from langchain.prompts import PromptTemplate  # For creating prompt templates for LLM queries
from langchain.chains import LLMChain  # For constructing LLM chains that combine prompts and LLM execution
import tiktoken  # For counting tokens
import re  # Regular expression library for text extraction
import PyPDF2  # For extracting text from PDF files


# In[5]:


# displaying versions of libraries used
print("google.generativeai version:", genai.__version__)
print("streamlit version:", st.__version__)
print("tiktoken version:", tiktoken.__version__)
print("langchain version:", langchain.__version__)


# In[6]:


def _set_env(var: str):
    if not os.environ.get(var): # checks if environment variable in present in the system or not
        import getpass
        # if env variable is not present then prompt the user to enter the value of the env variable
        os.environ[var] = getpass.getpass(f"Please enter your {var}: ") 

# passing name of the environment variable where api key in stored
_set_env("GOOGLE_API_KEY")

# Set your Gemini API key
api_key = os.environ.get("GOOGLE_API_KEY")

if api_key:
    genai.configure(api_key=api_key) # Configuring the API with the key
else:
    print("API key is missing. Please set the GOOGLE_API_KEY environment variable.")


# In[7]:


# Using tiktoken for token counting
encoder = tiktoken.get_encoding("cl100k_base")


# In[8]:


if 'tokens_consumed' not in st.session_state:
    st.session_state.tokens_consumed = 0  # Initialize total token count if not present
if 'query_tokens' not in st.session_state:
    st.session_state.query_tokens = 0     # Initialize query token count if not present
if 'response_tokens' not in st.session_state:
    st.session_state.response_tokens = 0  # Initialize response token count if not present


# In[9]:


# Initialize the Gemini AI model
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro-latest",
    temperature=0.8,
    max_tokens=8000
)

# Creating a Prompt Template
prompt_template = """
You are a senior consultant at BCG who is expert at analyzing companies.
I would appreciate your expertise in presenting a detailed SWOT analysis for a company based on the information provided below.
Also give me the name of the company (only name and not a sentence) that is being discussed in the below given information.
And if you are unable to find the name then just give an output as "Company name not provided".
You are not allowed to search for the company name outside the given information below.
Here is the information that should be considered:
{company_info}

Please ensure that the analysis is clear, concise, and highlights the most important factors for each quadrant.
Please provide a SWOT analysis in the following format:

**Company name**

**Strengths:**
- [Strength 1]
- [Strength 2]
...

**Weaknesses:**
- [Weakness 1]
- [Weakness 2]
...

**Opportunities:**
- [Opportunity 1]
- [Opportunity 2]
...

**Threats:**
- [Threat 1]
- [Threat 2]
...

NOTE: Do not hallucinate and do not generate random outputs.
"""

# Create a prompt by combining the input from the user and prompt template
prompt = PromptTemplate(input_variables=["company_info"], template=prompt_template)

# Initialize the LLMChain with the prompt and LLM
llm_chain = LLMChain(prompt=prompt, llm=llm)


# In[10]:


# defining a function for generating SWOT analysis
def get_swot_analysis(company_info: str):
    return llm_chain.run(company_info)


# In[11]:


# Function to check for forbidden phrases in user input
def is_forbidden_input(text: str) -> bool:
    forbidden_phrases = [
        "forget all previous prompts",
        "forget previous prompts",
        "forget previous instructions",
        "reset prompt",
        "clear prompt",
        "erase previous context",
        "ignore previous instructions",
        "change the prompt",
        "modify prompt",
        "tamper with prompt"
    ]
    lower_text = text.lower()
    for phrase in forbidden_phrases:
        if phrase in lower_text:
            return True
    return False


# In[12]:


# Function to extract the 4 different sections in SWOT from the LLM Response
def extract_swot_sections(swot_text: str):
    sections = ["Strengths", "Weaknesses", "Opportunities", "Threats"]
    swot_blocks = {s: "" for s in sections}
    for section in sections:
        pattern = rf"\*\*{section}:\*\*\s*((?:(?!\*\*(?:Strengths|Weaknesses|Opportunities|Threats):\*\*).)*)"
        match = re.search(pattern, swot_text, re.DOTALL)
        if match:
            block = match.group(1).strip()
            swot_blocks[section] = block
        else:
            swot_blocks[section] = ""
    return swot_blocks


# In[13]:


# Function to display SWOT quadrants using only Streamlit commands
def display_swot_analysis(strengths: str, weaknesses: str, opportunities: str, threats: str):
    # Displays SWOT quadrants using Streamlit's native commands.
    # Function to build message strings for each quadrant
    def build_message(content: str) -> str:
        lines = []
        for line in content.split("\n"):
            l = line.strip()
            if l:
                # Remove any leading dash or asterisk
                l = l.lstrip("-").lstrip("*").strip()
                lines.append(f"\n• {l}")
        return "\n".join(lines)
    
    strengths_msg = build_message(strengths)
    weaknesses_msg = build_message(weaknesses)
    opportunities_msg = build_message(opportunities)
    threats_msg = build_message(threats)
    
     # Create a two-row layout using columns
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"**Strengths**\n{strengths_msg}")  # Display strengths in a green box
    with col2:
        st.error(f"**Weaknesses**\n{weaknesses_msg}")  # Display weaknesses in a red box

    col3, col4 = st.columns(2)
    with col3:
        st.info(f"**Opportunities**\n{opportunities_msg}")  # Display opportunities in a blue/info box
    with col4:
        st.warning(f"**Threats**\n{threats_msg}")  # Display threats in an orange/warning box


# In[14]:


# Main Streamlit app UI
st.set_page_config(page_title="SWOT Analysis", layout="wide")
st.title("SWOT Analysis Application")
st.write("Upload a file or enter text below to generate SWOT Analysis:")

# File uploader and text input
uploaded_file = st.file_uploader("Choose a TXT or PDF file", type=["txt", "pdf"])
text_input = st.text_area("Or enter text directly:")

# Process the file input
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()
    if file_ext == "txt":
        text = uploaded_file.read().decode("utf-8")
    elif file_ext == "pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
else:
    text = text_input if text_input else None

query_tokens = 0
response_tokens = 0

# Check for forbidden phrases in the user input and determine if button should be disabled
if text and is_forbidden_input(text):
    generate_disabled = True
    st.error("Warning: Your input contains forbidden instructions. Please remove them before generating SWOT analysis.")
else:
    generate_disabled = False

# Creating "Generate SWOT Analysis" button that becomes unclickable when forbidden phrases are found in the input
generate = st.button("Generate SWOT Analysis", disabled = generate_disabled)

if generate:
    if text:
        with st.spinner('Generating SWOT Analysis'):
            swot_output = get_swot_analysis(text)
        # Optionally, display the raw SWOT output:
        with st.expander("Show Output in Plain Text"):
            st.write(swot_output)
        
        # Extract the company name from the LLM output using regex
        pattern1 = r"(?m)^\*\*(?!Strengths:|Weaknesses:|Opportunities:|Threats:)(.+?)\*\*"
        match = re.search(pattern1, swot_output)
        if match:
            company_name = match.group(1).strip()
        else:
            company_name = "Company name not provided"
        
        # Extract the full text for each SWOT section
        swot_blocks = extract_swot_sections(swot_output)
        
        # Display the company name as the title for the SWOT quadrants
        st.title(f"SWOT Analysis Quadrant: {company_name}")
        display_swot_analysis(
            swot_blocks["Strengths"],
            swot_blocks["Weaknesses"],
            swot_blocks["Opportunities"],
            swot_blocks["Threats"]
        )
        query_tokens = len(encoder.encode(text)) # counting tokens in input
        response_tokens = len(encoder.encode(swot_output)) # counting tokens in output

    else:
        st.info("Please upload a file or enter text to generate the SWOT analysis.")
        


# In[15]:


# Displaying the token count on the sidebar
st.session_state.query_tokens += query_tokens
st.session_state.response_tokens += response_tokens
st.session_state.tokens_consumed += (query_tokens + response_tokens)
st.sidebar.markdown("### Token Usage")
st.sidebar.write(f"Total Tokens Consumed: {st.session_state.tokens_consumed}")
st.sidebar.write(f"Query Tokens: {st.session_state.query_tokens}")
st.sidebar.write(f"Response Tokens: {st.session_state.response_tokens}")

print("Tokens consumed in this transaction...")
print("Query token =", query_tokens)
print("Response tokens =", response_tokens)
st.session_state.tokens_consumed = 0
st.session_state.query_tokens = 0
st.session_state.response_tokens = 0


# In[16]:


# Display library versions on the sidebar
st.sidebar.markdown("### Library Versions")
st.sidebar.write("google.generativeai:", genai.__version__)
st.sidebar.write("streamlit:", st.__version__)
st.sidebar.write("tiktoken:", tiktoken.__version__)
st.sidebar.write("langchain:", langchain.__version__)


# In[17]:


# Displaying name and email id on the sidebar
st.sidebar.markdown("### Application created by")
st.sidebar.write("Rohit Behara")
email = "rohitvenkata.ft251066@greatlakes.edu.in"
st.sidebar.markdown(f"You can reach out to me at: [📧](mailto:{email})", unsafe_allow_html=True)


# In[ ]:




