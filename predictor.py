import time
import aiosqlite
import traceback
import argparse
import asyncio
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

from mcp_agent.human_input.console_handler import console_input_callback
from mcp_agent.app import MCPApp

from utils import read_text, write_text

app = MCPApp(name="predictor-agent")

class PredictionResponse(BaseModel):
    prediction: bool
    reason: str 

async def runPredictor(instruction, prompts, actual):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name="predictor-agent",
            instruction=instruction,
            server_names=[],
        )
        convertor_agent = Agent(
            name="convertor-agent",
            instruction="""
                You are a data convertor agent. 

                Your task is to generate a structured response from the unstructured output of another AI agent.

                Follow the response model provided:
                prediction: Boolean value indicating the agent's prediction (success or not)
                reason: String value indicating the agent's explanation for the prediction

                Use only the information you are fed. Set the field to be an empty string if you find nothing in the output matching it. Do not leave it as `None`.
            """,
            server_names=[]
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            convertor_llm = await convertor_agent.attach_llm(GoogleAugmentedLLM)
            results = []
            for prompt in prompts:
                try:
                    result = await llm.generate_str(
                        message=prompt,
                        request_params=RequestParams(
                            max_iterations=20
                        ),
                    )
                    
                    try:
                        converted_result = await convertor_llm.generate_structured(
                            message=result,
                            response_model=PredictionResponse, 
                        )
                        results.append(converted_result)
                    except Exception as e:
                        print(f"Error converting result: {e}")
                        traceback.print_exc()
                        # Fallback or skip
                        # Creating a dummy failure response to avoid crashing the whole batch
                        results.append(PredictionResponse(
                            prediction=False,
                            reason=f"Error during conversion: {str(e)}",
                        ))
                        
                except Exception as e:
                    print(f"Error generating prediction: {e}")
                    traceback.print_exc()
                    results.append(PredictionResponse(
                        prediction=False,
                        reason=f"Error during generation: {str(e)}",
                    ))

            blocks = []
            tp = 0
            fp = 0
            tn = 0
            fn = 0
            for i, r in enumerate(results, 1):
                if (actual[i-1] == "1"):
                    if (r.prediction):
                        tp += 1
                    else:
                        fn += 1
                else:
                    if (r.prediction):
                        fp += 1
                    else:
                        tn += 1
                block = [
                        f"Prediction {i}:",
                        f"Agent answer: {r.prediction}",
                        f"Correct answer: {str(actual[i-1] == "1")}",
                        f"Reasoning: {r.reason}",
                        ""
                    ]
                blocks.append("\n".join(block))
            precision = 0
            recall = 0
            accuracy = 0
            fscore = 0
            if (tp+fp):
                precision = tp/(tp+fp)
            if (tp+fn):
                recall = tp/(tp+fn)
            if (tp+fn+tn+fp):
                accuracy = (tp+tn)/(tp+fn+tn+fp)
            if (tp+fn+fp):
                fscore = (1.25*tp)/(1.25*tp+0.25*fn+fp)
            return (f"**REPORT OF RESULTS:**\n\nF_0.5 score: {fscore}\nPrecision: {precision}\nRecall: {recall}\nAccuracy: {accuracy}\n\n") + ("\n".join(blocks))

async def uploadReport(text):
    async with aiosqlite.connect("files.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL
            )
        """)

        async with db.execute("SELECT COUNT(*) FROM reports") as cur:
            (count,) = await cur.fetchone()
        if count == 0:
            await db.execute("DELETE FROM sqlite_sequence WHERE name='reports'")
            await db.commit()

        await db.execute("""
            INSERT INTO reports (content, created_at)
            VALUES (?, ?)
        """, (text, datetime.now(timezone(timedelta(hours=7)))))
        await db.commit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prediction Agent script")
    parser.add_argument("--p",nargs="+",help="List of prompts to provide agent")
    parser.add_argument("--a",nargs="+",help="Actual results")

    args = parser.parse_args()
    prompts = args.p
    actual = args.a
    instructions = read_text("instructions.txt")
    
    report = asyncio.run(runPredictor(instructions, prompts, actual))
    write_text("report.txt", report)