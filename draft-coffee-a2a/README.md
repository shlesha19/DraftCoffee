# draft-coffee-mantha

**MANTHA** — General Data Pipeline Agent for the [A2A (Agent2Agent)](https://github.com/google-deepmind/a2a) network.

## What it does

MANTHA is an end-to-end data pipeline agent that can:

| Step | Tool | Description |
|---|---|---|
| 1 | `fetch_data` | Load and validate CSV / Excel files |
| 2 | `transform_data` | Clean, normalise, and type-coerce data (LLM-assisted) |
| 3 | `categorise_data` | LLM-assign categories to any column |
| 4 | `plot_data` | Auto-generate bar, line, pie, heatmap, histogram charts |
| 5 | `generate_report` | Build a formatted PDF report with charts and tables |
| 6 | `send_report` | Email the PDF via Gmail (API or SMTP fallback) |
| 7 | `run_full_pipeline` | Run all steps in one call |

## Quick Start

### Local

```bash
pip install -e .
export OPENAI_API_KEY=your-key
export OPENROUTER_API_KEY=your-key   # used by transformer & categorizer
python -m src --host localhost --port 5000
```

### Docker

```bash
# Edit docker-compose.yml — fill in your API keys
docker-compose up
```

## Project Structure

```
draft-coffee-mantha/
├── src/
│   ├── __init__.py               # Package init
│   ├── __main__.py               # A2A server entry point
│   ├── openai_agent.py           # Agent config + system prompt
│   ├── openai_agent_executor.py  # A2A OpenAI executor (from template)
│   ├── mantha_toolset.py         # All 7 pipeline tools as async methods
│   ├── fetcher.py                # CSV/Excel loader
│   ├── transformer.py            # Data cleaner + type coercer
│   ├── categorizer.py            # LLM categoriser (OpenRouter)
│   ├── plotter.py                # Chart generator (Matplotlib/Seaborn)
│   ├── report.py                 # PDF report builder (ReportLab)
│   ├── mail.py                   # Gmail sender (API + SMTP fallback)
│   └── pipeline.py               # CLI pipeline runner
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | ✅ Yes | Powers the A2A agent executor |
| `OPENROUTER_API_KEY` | ✅ Yes | Used by transformer & categorizer for LLM calls |
| `GMAIL_USER` | Optional | Gmail address for SMTP fallback |
| `GMAIL_APP_PASSWORD` | Optional | Gmail App Password for SMTP fallback |

## Example Queries

```
"Analyse my sales data from sales_q3.csv and send a report to team@company.com"
"Load data.xlsx, categorise the product column, and generate a PDF report"
"Run the full pipeline on orders.csv and email results to manager@company.com"
"Plot the data in report.csv and show me the charts"
```

## Testing

```bash
# Start the agent
python -m src --host localhost --port 5000

# Test with curl
curl -X POST http://localhost:5000/agent/message \
     -H "Content-Type: application/json" \
     -d '{"message": "Run the full pipeline on sample.csv"}'
```
