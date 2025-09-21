import os
from dotenv import load_dotenv
from galileo.experiments import run_experiment
from galileo.datasets import get_dataset
from galileo.openai import openai
from galileo.schema.metrics import GalileoScorers

load_dotenv()

client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
dataset = get_dataset(name="onboarding-project-dataset-1")

def llm_call(input):
    return client.chat.completions.create(
        model="gpt-4o",
        messages=[
          {
            "role": "system",
            "content": "You are an ecommerce shopping assistant for a keyboard company."
          },
          {
            "role": "user",
            "content": f"""This is the customer's question: {input}"""
          }
        ],
    ).choices[0].message.content

results = run_experiment(
    "test_experiment",
    dataset=dataset,
    function=llm_call,
    metrics=[GalileoScorers.correctness],
    project="onboarding-project"
)

print("Experiment successfully run and pushed to Galileo console")