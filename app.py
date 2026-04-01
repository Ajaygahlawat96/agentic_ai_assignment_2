import os
import streamlit as st
from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import WikipediaAPIWrapper

# Load environment variables
load_dotenv()

# Streamlit UI config
st.set_page_config(page_title="Autonomous Research Agent", page_icon="🤖")
st.title("🤖 Autonomous Research Agent (Groq + LangChain)")
st.write("Enter a topic and let the AI research it automatically!")

# Input field
topic = st.text_input("🔍 Enter Research Topic")

# Button
if st.button("Generate Report"):

    if not topic:
        st.warning("Please enter a topic!")
    else:
        with st.spinner("Researching... please wait ⏳"):

            # Get API key
            api_key = os.getenv("GROQ_API_KEY")

            # Initialize Groq LLM (OpenAI-compatible)
            llm = ChatOpenAI(
                openai_api_key=api_key,
                openai_api_base="https://api.groq.com/openai/v1",
                model_name="llama-3.3-70b-versatile",
                temperature=0
            )

            # Tools
            search = DuckDuckGoSearchRun()
            wiki = WikipediaAPIWrapper()

            tools = [
                Tool(
                    name="Web Search",
                    func=search.run,
                    description="Search latest information from the web"
                ),
                Tool(
                    name="Wikipedia",
                    func=wiki.run,
                    description="Get factual information from Wikipedia"
                )
            ]

            # Initialize ReAct Agent
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True
            )

            # Prompt
            prompt = f"""
            Research the topic: {topic}

            Generate a detailed structured report with:

            1. Cover Page
            2. Title
            3. Introduction
            4. Key Findings
            5. Challenges
            6. Future Scope
            7. Conclusion
            """

            # Run agent
            result = agent.run(prompt)

        # Display output
        st.success("Report Generated!")
        st.markdown(result)
