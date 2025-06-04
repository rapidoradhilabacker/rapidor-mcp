# MCP-based E-commerce Analytics Chatbot

This project is an end-to-end analytics solution for e-commerce platforms, featuring:

1. **Backend MCP Server (`server.py`)**
   - Exposes tools to fetch orders, invoices, attendance, sales metrics, and user sales details via MCP protocol.
   - Connects to multiple customer databases using credentials from environment variables.
   - Handles robust error reporting and reconnection logic.

2. **Chatbot Client (`chatbot.py`)**
   - Built with Streamlit for an interactive web-based chat interface.
   - Supports both **OpenAI** and **Google Gemini** LLM providers (select via the `LLM_PROVIDER` env var).
   - Parses user queries, extracts intent/parameters, and calls the backend tools.
   - Dynamic formatting for all MCP tools and improved error handling.

---

## Setup & Running the Project

### 1. Prerequisites
- Python 3.9 or higher
- Access to your e-commerce Postgres databases
- API key for your preferred LLM provider (OpenAI or Gemini)

### 2. Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd rapidor-mcp
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure environment variables:**
   - Copy `.env.example` to `.env` and set:
     - Database: `MASTER_DB_NAME`, `MASTER_DB_USER`, `MASTER_DB_PASSWORD`, `MASTER_DB_HOST`, `MASTER_DB_PORT`, `INITIALIZED_DBS`
     - LLM: `OPENAI_API_KEY` **or** `GEMINI_API_KEY`
     - LLM selection: `LLM_PROVIDER` (set to `openai` or `gemini`)

**A. Start the Streamlit Chatbot UI**
```bash
streamlit run chatbot.py --server.port 8080
```
This launches a browser-based chat interface (typically at http://localhost:8501).

### Usage
- Ask analytics questions in natural language (e.g., "Show total sales for john in acmeco.com from 2025-01-01 to 2025-01-31").
- The chatbot will extract intent, call the backend, and present the results.
- Supports queries for orders, invoices, attendance, leave, total sales, and user sales details.

#### Example Queries
- How many orders did user 'antim' have in testdemo7.rapidor.co between 2022-03-01 and 2022-03-10?
- Show me total sales for user 'john' in domain 'acmeco.com' from 2025-01-01 to 2025-01-31.
- List attendance for user 'susan' in domain 'example.com' for May 2025.
- Show Users and their sales details in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30.

---

## Troubleshooting
- **OpenAI/Gemini errors:** Check your API key, quota, and `LLM_PROVIDER` setting.
- **Database errors:** Verify credentials, database status, and that `INITIALIZED_DBS` contains the correct DB names.
- **Streamlit issues:** Ensure the MCP server is running and that there are no port conflicts.
- **No databases found:** Check your `.env` configuration and that the backend server can access the master database.
- **General errors:** Review terminal output for detailed logs.

---

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.

---

## License
MIT License. See `LICENSE` file for details.

- Enter your analytics questions in natural language, specifying the user, domain, and date range as needed.
- The chatbot will parse your request, query the backend, and display the results instantly.

#### Example Queries
- How many orders did user 'antim' have in testdemo7.rapidor.co between 2022-03-01 and 2022-03-10?
- Show me total sales for user 'john' in domain 'acmeco.com' from 2025-01-01 to 2025-01-31.
- List attendance for user 'susan' in domain 'example.com' for May 2025.

---

## Troubleshooting
- If you see OpenAI errors, check your API key and quota.
- For database errors, verify your credentials and database server status.
- If Streamlit does not load, ensure the MCP server is running and there are no port conflicts.

---

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.

---

## License
MIT License. See `LICENSE` file for details.
