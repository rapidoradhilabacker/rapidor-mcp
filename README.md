# MCP-based E-commerce Analytics Chatbot

This project contains two components:

1. **Backend MCP Server (`server.py`)**
   - Unchanged from before: exposes tools to fetch orders, invoices, attendance, and sales metrics via MCP.  
2. **Chatbot Client (`chatbot.py`)**
   - Uses Streamlit to present a chat interface.  
   - Routes user messages through OpenAI to parse intent and parameters, then invokes the MCP server.

---

## How to Run the Project

### 1. Prerequisites
- Python 3.9 or higher
- Access to your e-commerce databases
- An OpenAI API key

### 2. Installation Steps
1. **Clone this repository** to your local machine.
2. **Create a virtual environment** and activate it.
3. **Install dependencies** using the provided requirements file.
4. **Copy the `.env.example` file to `.env`** and fill in all required environment variables:
    - Database credentials (`MASTER_DB_NAME`, `MASTER_DB_USER`, `MASTER_DB_PASSWORD`, `MASTER_DB_HOST`, `MASTER_DB_PORT`)
    - Your `OPENAI_API_KEY`

### 3. Starting the Application
1. **Start the backend MCP server** by running the server script. This will make the analytics endpoints available.
2. **Start the chatbot UI** using Streamlit. This will launch a web interface for users to interact with the analytics chatbot.
3. **Open your browser** and navigate to the Streamlit URL (usually http://localhost:8501) to use the chatbot.

### 4. Usage
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
