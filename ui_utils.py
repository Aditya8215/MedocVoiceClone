import streamlit as st

def local_css():
    st.markdown("""
        <style>
        /* Main theme colors */
        .stApp {
            background-color: #0E1117; /* Black */
            color: #FAFAFA;
        }
        .st-emotion-cache-18ni7ap, .st-emotion-cache-z5fcl4 {
            background-color: #161A25; /* Slightly lighter black for sidebar/main content */
        }
        .stButton>button {
            border: 2px solid #1E88E5; /* Blue */
            background-color: transparent;
            color: #1E88E5;
            padding: 0.5em 1em;
            border-radius: 0.5em;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #1E88E5;
            color: white;
        }
        /* Style for the delete button */
        .stButton>button[kind="primary"] {
            border: 2px solid #D32F2F;
            background-color: #D32F2F;
            color: white;
        }
        .stButton>button[kind="primary"]:hover {
            background-color: #B71C1C;
            border-color: #B71C1C;
        }
        .st-emotion-cache-1v0mbdj, .st-emotion-cache-1xarl3l {
            border: 1px solid #1E88E5;
        }
        h1, h2, h3 {
            color: #1E88E5;
        }
        .st-emotion-cache-16txtl3 {
            padding: 2rem 1rem 1rem;
        }
        </style>
    """, unsafe_allow_html=True)