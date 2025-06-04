import streamlit as st
import os
import json
import subprocess
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai
from fastmcp import Client
from mcp.types import TextContent, ImageContent, EmbeddedResource

load_dotenv(dotenv_path='.env')

# Select LLM provider
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
if LLM_PROVIDER == "openai":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
elif LLM_PROVIDER == "gemini":
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    client = None  # Gemini does not use the same client interface
else:
    raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

def call_llm(prompt):
    """Call the selected LLM with the given prompt and return the response text."""
    if LLM_PROVIDER == "openai":
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    elif LLM_PROVIDER == "gemini":
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

class AsyncMCPClient:
    def __init__(self, server_url="http://localhost:8087/sse"):
        self.server_url = server_url
        self._client = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._client = Client(self.server_url)
        await self._client.__aenter__()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._client:
            await self._client.__aexit__(exc_type, exc_val, exc_tb)
    
    async def list_tools(self):
        """List available tools"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return await self._client.list_tools()
    
    async def call_tool(self, tool_name, params):
        """Call MCP tool via fastmcp client"""
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            print(f"DEBUG: Calling tool '{tool_name}' with params: {params}")  # DEBUG
            result = await self._client.call_tool(tool_name, params)
            print(f"DEBUG: Raw MCP result type: {type(result)}")  # DEBUG
            print(f"DEBUG: Raw MCP result: {result}")  # DEBUG
            
            extracted = self._extract_content(result)
            print(f"DEBUG: Extracted content: {extracted}")  # DEBUG
            return extracted
        except Exception as e:
            print(f"DEBUG: MCP call exception: {e}")  # DEBUG
            raise Exception(f"MCP call failed: {e}")
    
    def _extract_content(self, result):
        """Extract content from MCP response types"""
        print(f"DEBUG: _extract_content input type: {type(result)}")  # DEBUG
        print(f"DEBUG: _extract_content input: {result}")  # DEBUG
        
        if isinstance(result, list):
            content_parts = []
            for item in result:
                if isinstance(item, ImageContent):
                    content_parts.append(item.data)
                elif isinstance(item, EmbeddedResource):
                    content_parts.append(item.resource)
                elif isinstance(item, TextContent):
                    content_parts.append(item.text)
                else:
                    # Handle other types
                    content_parts.append(str(item))
            
            final_content = '\n'.join(content_parts) if content_parts else str(result)
            print(f"DEBUG: Final extracted content: {final_content}")  # DEBUG
            return final_content
        
        return str(result)

# Wrapper function to run async operations in Streamlit
def run_async(coro):
    """Run async coroutine in Streamlit"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def call_mcp_tool_sync(tool_name, db_name, username, start_date, end_date):
    """Synchronous wrapper for MCP tool calls"""
    async def _call_mcp_tool():
        try:
            async with AsyncMCPClient() as mcp_client:
                params = {
                    "db_name": db_name,
                    "username": username,
                    "start_date": start_date,
                    "end_date": end_date,
                }
                
                print(f"DEBUG: About to call MCP tool with params: {params}")  # DEBUG
                result = await mcp_client.call_tool(tool_name, params)
                print(f"DEBUG: MCP tool returned: {result}")  # DEBUG
                
                # Parse result if it's a JSON string
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        print(f"DEBUG: Successfully parsed JSON: {parsed}")  # DEBUG
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON parse error: {e}")  # DEBUG
                        return result
                
                return result
                
        except Exception as e:
            error_msg = f"Error calling MCP tool: {e}"
            print(f"DEBUG: {error_msg}")  # DEBUG
            return error_msg
    
    return run_async(_call_mcp_tool())

def build_prompt(user_message, db_options):
    """Build prompt for OpenAI to parse user intent"""
    db_list_str = "\n".join([f"- {d}" for d in db_options])
    prompt = f"""
You are an e-commerce analytics assistant. Users will ask about total orders, invoices, attendance, or sales for a sales user in a given date range. 

Available databases (by domain name) are:
{db_list_str}

When a user asks a question, extract:
- `action`: one of [get_total_orders, get_total_invoices, get_attendance, calculate_sales]
- `domain`: the customer domain they refer to (must match one of the listed domains)
- `username`: sales username
- `start_date` and `end_date` in YYYY-MM-DD format

If the user doesn't specify a domain, prompt them to choose from the list.
If dates are not specified, ask for clarification.

Respond in JSON format exactly with keys: action, domain, username, start_date, end_date.
Do not wrap the JSON in markdown code blocks - return only raw JSON.
If you cannot parse the request completely, respond with an error message explaining what's missing.

User query: "{user_message}"
"""
    return prompt

def format_result(result, action, username, start_date, end_date, domain):
    """Format the result for display"""
    print(f"DEBUG: format_result called with result: {result}, type: {type(result)}")  # DEBUG
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Handle case where result might still be the params dict
    if isinstance(result, dict) and 'action' in result:
        return f"❌ Error: Received parameters instead of data: {result}"
    
    if action == "get_total_orders":
        return f"📊 **Total Orders**: {result} orders created by {username} in {domain} from {start_date} to {end_date}"
    elif action == "get_total_invoices":
        return f"📋 **Total Invoices**: {result} invoices created by {username} in {domain} from {start_date} to {end_date}"
    elif action == "calculate_sales":
        return f"💰 **Total Sales**: ${result:,.2f} in sales by {username} in {domain} from {start_date} to {end_date}"
    elif action == "get_attendance":
        if isinstance(result, list) and result:
            attendance_summary = f"👤 **Attendance for {username}** from {start_date} to {end_date}:\n\n"
            for record in result:
                date = record.get('attendance_date', 'N/A')
                status = record.get('status', 'N/A')
                check_in = record.get('check_in', 'N/A')
                check_out = record.get('check_out', 'N/A')
                attendance_summary += f"• **{date}**: {status} (In: {check_in}, Out: {check_out})\n"
            return attendance_summary
        else:
            return f"No attendance records found for {username} in the specified date range."
    
    return str(result)

def check_mcp_connection():
    """Check MCP server connection and transform response content types."""
    async def _check_connection():
        try:
            async with AsyncMCPClient() as mcp_client:
                # Try to list tools as a connection test
                tools = await mcp_client.list_tools()
                print(f"DEBUG: Available tools: {tools}")  # DEBUG
                # Also try to fetch databases
                result = await mcp_client.call_tool("list_databases", {})
                return True, result
        except Exception as e:
            print(f"DEBUG: Connection check failed: {e}")  # DEBUG
            return False, str(e)
    
    return run_async(_check_connection())

# Streamlit UI
st.set_page_config(
    page_title="E-commerce Analytics Chatbot", 
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-commerce Analytics Chatbot")
st.markdown("Ask me about orders, invoices, attendance, or sales data for your team members!")

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Check MCP server connection
try:
    connected, db_result = check_mcp_connection()
    print(f"DEBUG: Connection status: {connected}, db_result: {db_result}")  # DEBUG
except Exception as e:
    connected, db_result = False, str(e)
    print(f"DEBUG: Connection exception: {e}")  # DEBUG

if not connected:
    st.error(f"⚠️ Cannot connect to MCP server: {db_result}")
    st.info("Make sure your MCP server is running and accessible via SSE at the configured URL")
    
    # Add information about server setup
    with st.expander("🔧 Server Setup Instructions"):
        st.code("""
# Make sure your MCP server supports SSE endpoint
# Example server URL: http://localhost:8087/sse
# If using a Python script as MCP server:
# You can also connect directly to the script file
# async with Client("server.py") as client:
#     ...
        """)
    
    # Add button to start server (if applicable)
    if st.button("🚀 Start MCP Server"):
        try:
            # Start server in background
            subprocess.Popen(["python", "server.py"])
            st.success("Server starting... Please refresh the page in a few seconds.")
        except Exception as e:
            st.error(f"Failed to start server: {e}")
    st.stop()

# Load databases
db_map = {}

def parse_mcp_result(db_result):
    """Parse MCP result to extract database information"""
    print(f"DEBUG: parse_mcp_result input: {db_result}, type: {type(db_result)}")  # DEBUG
    
    if isinstance(db_result, str):
        try:
            parsed = json.loads(db_result)
            print(f"DEBUG: Successfully parsed string as JSON: {parsed}")  # DEBUG
            return parsed
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse string as JSON: {e}")  # DEBUG
            return None
    elif hasattr(db_result, 'content') and db_result.content:
        # Handle MCP response types
        if isinstance(db_result.content, list):
            for item in db_result.content:
                if hasattr(item, 'text'):
                    try:
                        parsed = json.loads(item.text)
                        print(f"DEBUG: Successfully parsed content text as JSON: {parsed}")  # DEBUG
                        return parsed
                    except json.JSONDecodeError:
                        continue
        return None
    elif hasattr(db_result, 'text'):
        try:
            parsed = json.loads(db_result.text)
            print(f"DEBUG: Successfully parsed text attribute as JSON: {parsed}")  # DEBUG
            return parsed
        except json.JSONDecodeError:
            return None
    
    print(f"DEBUG: Returning db_result as-is: {db_result}")  # DEBUG
    return db_result

if connected:
    parsed_result = parse_mcp_result(db_result)
    print(f"DEBUG: Parsed database result: {parsed_result}")  # DEBUG
    
    if isinstance(parsed_result, list):
        try:
            db_map = {row['domain']: row['db_name'] for row in parsed_result if isinstance(row, dict) and 'domain' in row and 'db_name' in row}
            print(f"DEBUG: Created db_map: {db_map}")  # DEBUG
        except (KeyError, TypeError) as e:
            st.error(f"Error parsing database list: {e}")
            db_map = {}

if not db_map:
    st.warning("⚠️ No databases configured. Please check your database setup.")
    st.stop()

# Sidebar with available domains
with st.sidebar:
    st.header("🌐 Available Domains")
    for domain in db_map.keys():
        st.write(f"• {domain}")
    
    st.header("📋 Supported Queries")
    st.write("• Total orders for [username]")
    st.write("• Total invoices for [username]")
    st.write("• Attendance for [username]")
    st.write("• Sales data for [username]")
    st.write("• Date range: YYYY-MM-DD format")
    
    st.header("💡 Example Queries")
    st.code("Show total orders for john_doe in acme.com from 2024-01-01 to 2024-01-31")
    st.code("Get attendance for mary_smith in techcorp.com from 2024-02-01 to 2024-02-29")

# Chat interface
user_input = st.chat_input("Ask about your e-commerce data...")

if user_input:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})
    print(f"DEBUG: User input: {user_input}")  # DEBUG
    
    # Parse user intent with OpenAI
    prompt = build_prompt(user_input, list(db_map.keys()))
    
    try:
        content = call_llm(prompt)
        print(f"DEBUG: LLM response: {content}")  # DEBUG
        
        # Try to parse JSON response
        try:
            # Handle markdown code blocks
            clean_content = content.strip()
            if clean_content.startswith('```json'):
                # Extract JSON from markdown code block
                clean_content = clean_content[7:]  # Remove ```json
                if clean_content.endswith('```'):
                    clean_content = clean_content[:-3]  # Remove closing ```
                clean_content = clean_content.strip()
            elif clean_content.startswith('```'):
                # Handle generic code blocks
                lines = clean_content.split('\n')
                if len(lines) > 1:
                    clean_content = '\n'.join(lines[1:-1])  # Remove first and last line
                clean_content = clean_content.strip()
            
            print(f"DEBUG: Cleaned LLM content: {clean_content}")  # DEBUG
            params = json.loads(clean_content)
            print(f"DEBUG: Parsed LLM params: {params}")  # DEBUG
            
            action = params.get("action")
            domain = params.get("domain")
            username = params.get("username")
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            
            # Validate required parameters
            if not all([action, domain, username, start_date, end_date]):
                missing = [k for k, v in params.items() if not v]
                reply = f"❌ Missing required information: {', '.join(missing)}. Please provide all details."
            elif domain not in db_map:
                reply = f"❌ Unknown domain `{domain}`. Available domains: {', '.join(db_map.keys())}"
            else:
                # Call MCP tool
                db_name = db_map[domain]
                print(f"DEBUG: Using db_name: {db_name} for domain: {domain}")  # DEBUG
                
                with st.spinner(f"Fetching {action} data..."):
                    result = call_mcp_tool_sync(action, db_name, username, start_date, end_date)
                    print(f"DEBUG: Final result from MCP: {result}")  # DEBUG
                    reply = format_result(result, action, username, start_date, end_date, domain)
                
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")  # DEBUG
            reply = content  # OpenAI couldn't parse - likely an error message
            
    except Exception as e:
        error_msg = f"❌ Error processing request: {e}"
        print(f"DEBUG: Processing error: {e}")  # DEBUG
        reply = error_msg
    
    # Add assistant response to history
    st.session_state.history.append({"role": "assistant", "content": reply})

# Display chat history
for i, chat in enumerate(st.session_state.history):
    if chat['role'] == 'user':
        with st.chat_message("user"):
            st.write(chat['content'])
    else:
        with st.chat_message("assistant"):
            st.markdown(chat['content'])

# Clear chat button
if st.session_state.history:
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state.history = []
            st.rerun()

# Connection status in sidebar
with st.sidebar:
    st.markdown("---")
    if connected:
        st.success("✅ MCP Server Connected")
        st.caption(f"Found {len(db_map)} databases")
    else:
        st.error("❌ MCP Server Disconnected")

# Footer
st.markdown("---")
st.caption("💡 Example: 'Show total orders for john_doe in acme.com from 2024-01-01 to 2024-01-31'")