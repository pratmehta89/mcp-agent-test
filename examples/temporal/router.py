"""
Example of using Temporal as the execution engine for MCP Agent workflows.
This example demonstrates how to create a workflow using the app.workflow and app.workflow_run
decorators, and how to run it using the Temporal executor.
"""

import asyncio
import os

from mcp_agent.agents.agent import Agent
from mcp_agent.executor.temporal import TemporalExecutor
from mcp_agent.executor.workflow import Workflow, WorkflowResult
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.router.router_llm import LLMRouter
from mcp_agent.workflows.router.router_llm_anthropic import AnthropicLLMRouter

from main import app


def print_to_console(message: str):
    """
    A simple function that prints a message to the console.
    """
    print(message)


def print_hello_world():
    """
    A simple function that prints "Hello, world!" to the console.
    """
    print_to_console("Hello, world!")


@app.workflow
class RouterWorkflow(Workflow[str]):
    """
    A simple workflow that demonstrates the basic structure of a Temporal workflow.
    """

    @app.workflow_run
    async def run(self, input: str) -> WorkflowResult[str]:
        """
        Run the workflow, processing the input data.

        Args:
            input_data: The data to process

        Returns:
            A WorkflowResult containing the processed data
        """

        logger = app.logger
        context = app.context
        context.config.mcp.servers["filesystem"].args.extend([os.getcwd()])

        finder_agent = Agent(
            name="finder",
            instruction="""You are an agent with access to the filesystem, 
            as well as the ability to fetch URLs. Your job is to identify 
            the closest match to a user's request, make the appropriate tool calls, 
            and return the URI and CONTENTS of the closest match.""",
            server_names=["fetch", "filesystem"],
        )

        writer_agent = Agent(
            name="writer",
            instruction="""You are an agent that can write to the filesystem.
            You are tasked with taking the user's input, addressing it, and 
            writing the result to disk in the appropriate location.""",
            server_names=["filesystem"],
        )

        reasoning_agent = Agent(
            name="reasoner",
            instruction="""You are a generalist with knowledge about a vast
            breadth of subjects. You are tasked with analyzing and reasoning over
            the user's query and providing a thoughtful response.""",
            server_names=[],
        )

        # You can use any LLM with an LLMRouter
        llm = OpenAIAugmentedLLM(name="openai_router", instruction="You are a router")
        router = LLMRouter(
            llm=llm,
            agents=[finder_agent, writer_agent, reasoning_agent],
            functions=[print_to_console, print_hello_world],
            context=app.context,
        )

        # This should route the query to finder agent, and also give an explanation of its decision
        results = await router.route_to_agent(
            request="Print the contents of mcp_agent.config.yaml verbatim", top_k=1
        )

        logger.info("Router Results:", data=results)

        # We can use the agent returned by the router
        agent = results[0].result
        async with agent:
            result = await agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            result = await agent.call_tool(
                name="read_file",
                arguments={
                    "path": str(os.path.join(os.getcwd(), "mcp_agent.config.yaml"))
                },
            )
            logger.info("read_file result:", data=result.model_dump())

        # We can also use a router already configured with a particular LLM
        anthropic_router = AnthropicLLMRouter(
            server_names=["fetch", "filesystem"],
            agents=[finder_agent, writer_agent, reasoning_agent],
            functions=[print_to_console, print_hello_world],
            context=app.context,
        )

        # This should route the query to print_to_console function
        # Note that even though top_k is 2, it should only return print_to_console and not print_hello_world
        results = await anthropic_router.route_to_function(
            request="Print the input to console", top_k=2
        )
        logger.info("Router Results:", data=results)
        function_to_call = results[0].result
        function_to_call("Hello, world!")

        # This should route the query to fetch MCP server (inferring just by the server name alone!)
        # You can also specify a server description in mcp_agent.config.yaml to help the router make a more informed decision
        results = await anthropic_router.route_to_server(
            request="Print the first two paragraphs of https://modelcontextprotocol.io/introduction",
            top_k=1,
        )
        logger.info("Router Results:", data=results)

        # Using the 'route' function will return the top-k results across all categories the router was initialized with (servers, agents and callables)
        # top_k = 3 should likely print: 1. filesystem server, 2. finder agent and possibly 3. print_to_console function
        results = await anthropic_router.route(
            request="Print the contents of mcp_agent.config.yaml verbatim",
            top_k=3,
        )
        logger.info("Router Results:", data=results)

        return WorkflowResult(value="Success")


async def main():
    async with app.run() as orchestrator_app:
        executor: TemporalExecutor = orchestrator_app.executor

        handle = await executor.start_workflow(
            "RouterWorkflow",
            None,
        )
        a = await handle.result()
        print(a)


if __name__ == "__main__":
    asyncio.run(main())
