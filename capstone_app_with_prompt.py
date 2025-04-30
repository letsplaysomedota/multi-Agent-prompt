# app.py

import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime
from wordfilter import Wordfilter
import openai

# Load environment variables from your "env" file.
load_dotenv("env")

# Set the docker flag if needed by autogen.
os.environ['AUTOGEN_USE_DOCKER'] = '0'

# Import the autogen classes.
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Define the llm_config for DeepSeek
llm_config = {
    "model": "deepseek-chat",  
    "api_key": os.getenv("DEEPSEEK_API_KEY"),
    "base_url": os.getenv("DEEPSEEK_BASE_URL")
}

# Initialize a Wordfilter instance if used by the agents
wordfilter = Wordfilter()

# Define the agents
prompt_generator_agent = AssistantAgent(
    name="Prompt_Generator_Agent",
    system_message="""You are an agent that generates the initial prompt based on a broad description of the user's task.
Do:
-Provide a clear and concise initial prompt.
-Focus on capturing the essence of the task.
Don't:
-Overcomplicate the prompt with unnecessary details.
""",
    llm_config=llm_config  
)

clarity_agent = AssistantAgent(
    name="Clarity_Agent",
    system_message="""You are an agent that enhances the clarity and understandability of the prompt.
Do:
- Rephrase the prompt to be easily understood.
- Remove ambiguity.
Don't:
- Change the intended meaning.
""",
    llm_config=llm_config  
)

relevance_agent = AssistantAgent(
    name="Relevance_Agent",
    system_message="""You are an agent that ensures the prompt remains focused on the user's original intent.
Do:
- Keep the response on topic.
- Remove extraneous or irrelevant information.
Don't:
- Introduce unrelated details.
""",
    llm_config=llm_config  
)

precision_agent = AssistantAgent(
    name="Precision_Agent",
    system_message="""You are an agent that increases the precision of the prompt.
Do:
- Ask for or include specific details where necessary.
- Narrow down the scope to be more specific.
Don't:
- Provide vague or overly general responses.
""",
    llm_config=llm_config  
)

creativity_agent = AssistantAgent(
    name="Creativity_Agent",
    system_message="""You are an agent that encourages more imaginative and innovative outputs based on what would delight the end user.
Do:
- Introduce creative angles or analogies when appropriate.
- Enhance the prompt with creative suggestions.
Don't:
- Overcomplicate or confuse the prompt.
""",
    llm_config=llm_config  
)

completeness_agent = AssistantAgent(
    name="Completeness_Agent",
    system_message="""You are the Refinement Agent. Your role is to synthesize and refine the outputs from the previous agents into a final, concise prompt.
Do:
- Combine key information from each agent into a coherent, actionable prompt.
- Ensure that the final prompt captures all essential details.
Don't:
- Simply concatenate outputs without integration.
- Include irrelevant or redundant information.
""",
    llm_config=llm_config  
)

# ------------------------
# Define the QA Agent as the Improved Prompt Generator
# ------------------------

qa_agent = AssistantAgent(
    name="QA_Agent",
    system_message="""You are the QA Agent.
Your task is to take the user's original prompt along with the refined output from the other agents and generate an improved initial prompt.
This improved prompt should more accurately capture the user's intended request and be more likely to achieve a valuable output.
Do:
- Consider both the original prompt and the additional details provided by the other agents.
- Generate a revised version that is clear, focused, and actionable.
Don't:
- Simply echo the original prompt or include irrelevant details.
""",
    llm_config=llm_config  
)


# Define helper functions for the multi-agent pipeline

def run_parallel_agents(prompt):
    # Check for sensitive words if needed
    if wordfilter.blacklisted(prompt):
        st.error("The input contains sensitive words. Please rephrase your question.")
        return None

    responses = {}
    messages = [{"role": "user", "content": prompt}]

    # Get outputs from the primary agents
    responses[prompt_generator_agent.name] = prompt_generator_agent.generate_reply(messages)
    responses[clarity_agent.name] = clarity_agent.generate_reply(messages)
    responses[relevance_agent.name] = relevance_agent.generate_reply(messages)
    responses[precision_agent.name] = precision_agent.generate_reply(messages)
    responses[creativity_agent.name] = creativity_agent.generate_reply(messages)

    # Log agent outputs (optional print statements)
    for name, output in responses.items():
        print(f"{name} Output: {output}\n")

    # Combine responses and refine via the completeness agent
    combined_input = "\n".join([f"{name}: {resp}" for name, resp in responses.items()])
    final_messages = [{"role": "user", "content": combined_input}]
    refined_prompt = completeness_agent.generate_reply(final_messages)
    print("Final Refined Prompt:", refined_prompt)
    return refined_prompt

def run_qa_agent(initial_prompt, refined_prompt):
    qa_messages = [
        {"role": "user", "content": f"User's initial prompt: {initial_prompt}"},
        {"role": "user", "content": f"Refined details: {refined_prompt}"}
    ]
    improved_prompt = qa_agent.generate_reply(qa_messages)
    print("Improved Initial Prompt:", improved_prompt)
    return improved_prompt

# Streamlit UI for Testing

st.title("Multi-Agent Improved Prompt Generator")

user_input = st.text_input("Enter your task description:")

if st.button("Generate Improved Prompt"):
    refined_prompt = run_parallel_agents(user_input)
    if refined_prompt:
        improved_prompt = run_qa_agent(user_input, refined_prompt)
        st.session_state['improved_prompt'] = improved_prompt
        st.subheader("Improved Initial Prompt")
        st.write(improved_prompt)