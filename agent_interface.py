import re
from typing import List, Union

import chainlit as cl
from langchain import LLMChain
from langchain.agents import (
    AgentExecutor,
    AgentOutputParser,
    LLMSingleActionAgent,
    Tool,
)
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import BaseModel, Field

from common.text_processing.document_indexer import DocumentIndexer

chain_template = """You are an expert Python engineer.  You have access to the following tools:

{tools}

This is our previous conversation history:

{history}

I will ask a question, then you will think about what to do next, and then take an action. You will then observe the result of that action, and then think about what to do next. This will repeat until you have a detailed final answer to the original question. Always use the following format as your response:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a very detailed final answer to the original input question

Begin! Remember to answer as a expert Python engineer when giving your detailed final answer. And remember to prefix it with "Final Answer: ".

Question: {human_input}
{agent_scratchpad}"""


class CustomPromptTemplate(StringPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools]
        )
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)


class CustomOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        # Return the action and action input
        return AgentAction(
            tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output
        )

class KBInput(BaseModel):
    retriever: VectorStoreRetriever = Field()
    llm: ChatOpenAI = Field()

def search_langchain(input_text):
    search = DuckDuckGoSearchRun().run(
        f"site:https://python.langchain.com/ {input_text}"
    )
    return search

def search_general(input_text):
    search = DuckDuckGoSearchRun().run(f"{input_text}")
    return search


def search_knowledge(input_text):
    retriever = DocumentIndexer().get_vector_store().as_retriever()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048)
    qa = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa({"question": {input_text}})["answer"]


@cl.langchain_factory(use_async=False)
def agent():

    tools = [
        # Tool(
        #     name="Search knowledge base",
        #     func=search_knowledge,
        #     description="useful for all first attempts at answering questions",
        # ),
        Tool(
            name="Search langchain",
            func=search_langchain,
            description="useful for when you need to answer questions about langchain",
        ),
        Tool(
            name="Search general",
            func=search_general,
            description="useful for when you need to answer general questions not related to buddhism",
        ), 
    ]
    
    prompt = CustomPromptTemplate(
        template=chain_template,
        tools=tools,
        # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
        # This includes the `intermediate_steps` variable because that is needed
        input_variables=["human_input", "intermediate_steps", "history"],
    )

    output_parser = CustomOutputParser()
    memory = ConversationBufferMemory()
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, max_tokens=2048)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    tool_names = [tool.name for tool in tools]

    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:"],
        allowed_tools=tool_names,
    )

    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, memory=memory, verbose=True
    )
    return agent_executor
