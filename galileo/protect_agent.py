from galileo.openai import openai
from galileo.protect import invoke_protect
from galileo_core.schemas.protect.execution_status import ExecutionStatus
from galileo_core.schemas.protect.payload import Payload
from dotenv import load_dotenv

load_dotenv()

client = openai.OpenAI()

while True:
    # get input from the user
    user_input = input("User: ")

    # check user input to see if conversation should end
    if user_input.lower() in ["bye", "goodbye", ""]:
        break

    # create the payload
    payload = Payload(
        input=user_input
    )

    # invoke runtime protection
    protection_response = invoke_protect(
        stage_name = "Toxicity Stage",
        payload=payload
    )

    # check runtime protection status - if triggered, print action result, skip llm call, end convo
    if protection_response.status == ExecutionStatus.triggered:
        print(f"Assitant: {protection_response.action_result['value']}")
        break        

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": user_input}],
    )

    print(f"Assistant: {response.choices[0].message.content.strip()}")