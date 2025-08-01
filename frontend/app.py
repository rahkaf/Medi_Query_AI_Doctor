# mediquery/frontend/app.py
import streamlit as st
import requests
import json

# Initialize session state for history and quiz
if "history" not in st.session_state:
    st.session_state.history = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = None
if "current_page" not in st.session_state:
    st.session_state.current_page = "Main"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Main", "Quiz Generation", "Quiz Answering"], key="nav")
if page != st.session_state.current_page:
    st.session_state.history.append(page)
    st.session_state.current_page = page

# History sidebar
st.sidebar.title("History")
for hist_page in st.session_state.history[-5:]:  # Show last 5 pages
    st.sidebar.write(hist_page)

if page == "Main":
    st.title("MediQuery: Medical Information Retrieval")
    st.write("Ask questions about HIV/AIDS based on processed documents.")

    # Query section
    st.subheader("Ask a Question")
    query = st.text_input("Enter your question (e.g., 'Please write your question! :) ')")
    mode = st.selectbox("Select mode", ["quick", "research"])
    if st.button("Submit Query"):
        if query:
            with st.spinner("Processing query..."):
                try:
                    response = requests.post(
                        "http://localhost:8000/query/",  # Use 8001 if changed
                        headers={"Content-Type": "application/json"},
                        json={"query": query, "mode": mode}
                    )
                    response.raise_for_status()
                    result = response.json()
                    st.subheader("Answer")
                    st.write(result["answer"])
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to backend: {str(e)}")
        else:
            st.warning("Please enter a question.")

elif page == "Quiz Generation":
    st.title("MediQuery: Quiz Generation")
    st.write("Generate a quiz on HIV/AIDS to test your knowledge.")

    # Quiz generation form
    st.subheader("Generate Quiz")
    with st.form(key="quiz_form"):
        query = st.text_input("Enter a topic for the quiz (e.g., 'HIV transmission')")
        num_questions = st.number_input("Number of questions", min_value=1, max_value=10, value=5)
        submit_quiz = st.form_submit_button("Generate Quiz")

    if submit_quiz and query:
        with st.spinner("Generating quiz..."):
            try:
                response = requests.post(
                    "http://localhost:8000/query/",  # Use 8001 if changed
                    headers={"Content-Type": "application/json"},
                    json={"query": query, "mode": "quiz", "num_questions": num_questions}
                )
                response.raise_for_status()
                result = response.json()
                if isinstance(result["answer"], dict) and "questions" in result["answer"]:
                    st.session_state.quiz_questions = result["answer"]["questions"]
                    st.session_state.quiz_answers = [None] * len(result["answer"]["questions"])
                    st.success("Quiz generated! Navigate to 'Quiz Answering' to take the quiz.")
                else:
                    st.error(result["answer"])
            except requests.exceptions.RequestException as e:
                st.error(f"Error generating quiz: {str(e)}")

elif page == "Quiz Answering":
    st.title("MediQuery: Quiz Answering")
    st.write("Answer the generated quiz questions by selecting one option for each question.")

    if not st.session_state.quiz_questions:
        st.warning("No quiz generated. Please go to 'Quiz Generation' to create a quiz.")
    else:
        st.subheader("Quiz")
        correct_answers = 0
        total_questions = len(st.session_state.quiz_questions)
        
        # Create columns for better layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            for i, question in enumerate(st.session_state.quiz_questions):
                st.markdown(f"**Question {i+1}:** {question['question']}")
                
                # Create radio buttons for options
                answer = st.radio(
                    "Select your answer:",
                    options=question['options'],
                    key=f"q_{i}",
                    index=None  # No default selection
                )
                
                # Store the answer in session state
                if f"answer_{i}" not in st.session_state:
                    st.session_state[f"answer_{i}"] = None
                
                if answer:
                    st.session_state[f"answer_{i}"] = answer
                
                st.markdown("---")  # Add separator between questions
        
        # Submit button
        if st.button("Submit Quiz"):
            score = 0
            for i, question in enumerate(st.session_state.quiz_questions):
                if st.session_state.get(f"answer_{i}"):
                    selected_answer = question['options'].index(st.session_state[f"answer_{i}"])
                    if selected_answer == question['correct_answer']:
                        score += 1
            
            # Display results
            st.success(f"Quiz completed! Your score: {score}/{total_questions}")
            percentage = (score/total_questions) * 100
            st.progress(percentage/100)  # Show progress bar
            
            # Add a retry button
            if st.button("Try Another Quiz"):
                st.session_state.quiz_questions = None
                st.session_state.quiz_answers = None
                for i in range(total_questions):
                    if f"answer_{i}" in st.session_state:
                        del st.session_state[f"answer_{i}"]
                st.experimental_rerun()