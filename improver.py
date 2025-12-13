import time
import aiosqlite
from datetime import datetime, timezone, timedelta

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from mcp_agent.human_input.console_handler import console_input_callback
from mcp_agent.app import MCPApp

app = MCPApp(name="improver-agent")

async def runImprover(prompt):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name="improver-agent",
            instruction="""
            You are an expert prompt engineer. 

            You are trying to build the best prediction agent possible to predict "outlier successes" based on anonymised founder profiles.

            For context, this founder's profile is anonymized, with only details on educational and professional background, as well as previous IPOs and acquisitions. 

            The CSV file which is where the data is from contains the headers: founder_uuid,success,industry,ipos,acquisitions,educations_json,jobs_json,anonymised_prose
    An example row looks like:
    33159ebb-97ff-43fe-a80e-31fdcf467065,1,"Technology, Information & Internet Platforms",,,"[
  {
    ""degree"": ""BA"",
    ""field"": ""Computer Science"",
    ""qs_ranking"": ""1""
  }
]","[
  {
    ""role"": ""CTO"",
    ""company_size"": ""myself only employees"",
    ""industry"": ""Sports Teams & Leagues"",
    ""duration"": ""<2""
  },
  {
    ""role"": ""CTO"",
    ""company_size"": ""2-10 employees"",
    ""industry"": ""E-Learning"",
    ""duration"": ""<2""
  },
  {
    ""role"": ""Software Engineer"",
    ""company_size"": ""2-10 employees"",
    ""industry"": ""Wellness & Community Health"",
    ""duration"": ""<2""
  },
  {
    ""role"": ""Graduate Fellow (NSF)"",
    ""company_size"": ""201-500 employees"",
    ""industry"": ""Environmental & Waste Services"",
    ""duration"": ""4-5""
  }
]","This founder leads a startup in the Technology, Information & Internet Platforms industry.
Education:
* BA in Computer Science (Institution QS rank 1)

Professional experience:
* CTO for <2 years in the `Sports Teams & Leagues` industry (myself only employees)
* CTO for <2 years in the `E-Learning` industry (2-10 employees)
* Software Engineer for <2 years in the `Wellness & Community Health` industry (2-10 employees)
* Graduate Fellow (NSF) for 4-5 years in the `Environmental & Waste Services` industry (201-500 employees)
"

            You will be provided:
            1. the current instructions passed to the agent
            2. a F_0.5, precision, recall, and accuracy scoring of the agent, as well as its responses, the actual answers, alongside the agent's reasoning
            
            Your task is to write new instructions to pass to the agent.

            Make sure that the agent's output format is unchanged:
            1. prediction (True or Flase): Success or not
            2. reason: One paragraph reasoning for prediction 
            """,
            server_names=[],
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            # convertor_llm = await convertor_agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=20,
                    model="gemini-3-pro-preview"  # Set your desired limit
                ),
            )
            
            end = time.time()
            logger.info(f"[improver-agent] Worked for {end-start}")
            return result