import logging
import os

import click
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv
from openai_agent import create_agent  # type: ignore[import-not-found]
from openai_agent_executor import (
    OpenAIAgentExecutor,  # type: ignore[import-untyped]
)
from starlette.applications import Starlette


load_dotenv()

logging.basicConfig()


@click.command()
@click.option('--host', 'host', default='localhost')
@click.option('--port', 'port', default=5000)
def main(host: str, port: int):
    # Verify an API key is set.
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY environment variable not set')

    skill = AgentSkill(
        id='mantha_data_pipeline',
        name='MANTHA Data Pipeline',
        description=(
            'End-to-end data pipeline agent: loads CSV/Excel files, cleans and '
            'transforms data, LLM-categorises columns, auto-generates charts, '
            'builds a formatted PDF report, and emails it to recipients.'
        ),
        tags=['data', 'pipeline', 'csv', 'excel', 'analytics', 'report', 'email', 'llm'],
        examples=[
            'Analyse my sales data from sales_q3.csv and send a report to team@company.com',
            'Load data.xlsx, categorise the product column, and generate a PDF report',
            'Run the full pipeline on orders.csv and email results to manager@company.com',
            'Plot the data in report.csv and show me the charts',
            'Transform and clean my dataset in raw_data.csv',
        ],
    )

    # AgentCard for the MANTHA agent
    agent_card = AgentCard(
        name='draft-coffee-mantha',
        description=(
            'MANTHA is a general-purpose data pipeline agent. It can load CSV/Excel '
            'files, clean and transform data using LLM-assisted type inference, '
            'auto-categorise columns, generate publication-quality charts, produce '
            'formatted PDF reports, and deliver them via Gmail.'
        ),
        url=f'http://{host}:{port}/',
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    # Create MANTHA agent
    agent_data = create_agent()

    agent_executor = OpenAIAgentExecutor(
        card=agent_card,
        tools=agent_data['tools'],
        api_key=os.getenv('OPENAI_API_KEY'),
        system_prompt=agent_data['system_prompt'],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor, task_store=InMemoryTaskStore()
    )

    a2a_app = A2AStarletteApplication(
        agent_card=agent_card, http_handler=request_handler
    )
    routes = a2a_app.routes()

    app = Starlette(routes=routes)

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
