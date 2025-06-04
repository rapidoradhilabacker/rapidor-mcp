import streamlit as st
import os
import json
import subprocess
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import google.generativeai as genai

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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

class MCPClient:
    def __init__(self, server_script_path="server.py"):
        self.server_script_path = server_script_path
        self.session = None
        self.available_tools = []
        self.stdio_transport = None
        self.read_stream = None
        self.write_stream = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        # Create server parameters for stdio connection
        server_params = StdioServerParameters(
            command="python",
            args=[self.server_script_path],
            env=None
        )
        
        # Create stdio client and session
        self.stdio_transport = stdio_client(server_params)
        self.read_stream, self.write_stream = await self.stdio_transport.__aenter__()
        
        # Create client session
        self.session = ClientSession(self.read_stream, self.write_stream)
        await self.session.__aenter__()
        
        # Add retry logic for initialization
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Initialize the session
                await self.session.initialize()
                
                # Get available tools
                tools_result = await self.session.list_tools()
                self.available_tools = tools_result.tools
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(f"Failed to initialize MCP session after {max_retries} attempts: {e}")
                await asyncio.sleep(1)  # Wait before retry
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        try:
            if self.session:
                await self.session.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            print(f"DEBUG: Error closing session: {e}")
        
        try:
            if self.stdio_transport:
                await self.stdio_transport.__aexit__(exc_type, exc_val, exc_tb)
        except Exception as e:
            print(f"DEBUG: Error closing transport: {e}")
    
    async def list_tools(self):
        """List available tools"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        return self.available_tools
    
    async def call_tool(self, tool_name, arguments):
        """Call MCP tool via stdio client session"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            print(f"DEBUG: Calling tool '{tool_name}' with arguments: {arguments}")
            result = await self.session.call_tool(tool_name, arguments)
            print(f"DEBUG: Raw MCP result: {result}")
            
            extracted = self._extract_content(result)
            print(f"DEBUG: Extracted content: {extracted}")
            return extracted
        except Exception as e:
            print(f"DEBUG: MCP call exception: {e}")
            raise Exception(f"MCP call failed: {e}")
    
    def _extract_content(self, result):
        """Extract content from MCP response"""
        print(f"DEBUG: _extract_content input type: {type(result)}")
        print(f"DEBUG: _extract_content input: {result}")
        
        # Handle CallToolResult
        if hasattr(result, 'content') and result.content:
            content_parts = []
            for item in result.content:
                if hasattr(item, 'text'):
                    content_parts.append(item.text)
                elif hasattr(item, 'data'):
                    content_parts.append(str(item.data))
                else:
                    content_parts.append(str(item))
            
            final_content = '\n'.join(content_parts) if content_parts else str(result)
            print(f"DEBUG: Final extracted content: {final_content}")
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
            async with MCPClient() as mcp_client:
                arguments = {
                    "db_name": db_name,
                    "username": username,
                    "start_date": start_date,
                    "end_date": end_date,
                }
                
                # Remove username for tools that don't need it
                if tool_name == "get_users_and_sales_details":
                    arguments.pop("username", None)
                
                print(f"DEBUG: About to call MCP tool with arguments: {arguments}")
                result = await mcp_client.call_tool(tool_name, arguments)
                print(f"DEBUG: MCP tool returned: {result}")
                
                # Parse result if it's a JSON string
                if isinstance(result, str):
                    try:
                        parsed = json.loads(result)
                        print(f"DEBUG: Successfully parsed JSON: {parsed}")
                        return parsed
                    except json.JSONDecodeError as e:
                        print(f"DEBUG: JSON parse error: {e}")
                        return result
                
                return result
                
        except Exception as e:
            error_msg = f"Error calling MCP tool: {e}"
            print(f"DEBUG: {error_msg}")
            return error_msg
    
    return run_async(_call_mcp_tool())

def build_prompt(user_message, db_options, available_actions):
    """Build prompt for LLM to parse user intent"""
    db_list_str = "\n".join([f"- {d}" for d in db_options])
    prompt = f"""
You are an e-commerce analytics assistant. Users will ask about total orders, invoices, attendance, or sales for a sales user in a given date range. 

Available databases (by domain name) are:
{db_list_str}

Available actions are: {', '.join(available_actions)}

When a user asks a question, extract:
- `action`: one of {available_actions}
- `domain`: the customer domain they refer to (must match one of the listed domains)
- `username`: sales username (not required for get_users_and_sales_details)
- `start_date` and `end_date` in YYYY-MM-DD format

If the user doesn't specify a domain, prompt them to choose from the list.
If dates are not specified, ask for clarification.

Respond in JSON format exactly with keys: action, domain, username, start_date, end_date.
Do not wrap the JSON in markdown code blocks - return only raw JSON.
If you cannot parse the request completely, respond with an error message explaining what's missing.

User query: "{user_message}"
"""
    return prompt

def get_available_actions():
    async def _get_tools():
        async with MCPClient() as mcp_client:
            tools = await mcp_client.list_tools()
            # tools can be Tool objects or strings depending on implementation
            # If Tool objects, extract .name
            if tools and hasattr(tools[0], 'name'):
                return [tool.name for tool in tools]
            return tools
    return run_async(_get_tools())

def format_result(result, action, username, start_date, end_date, domain):
    """Format the result for display"""
    print(f"DEBUG: format_result called with result: {result}, type: {type(result)}")
    
    if isinstance(result, str) and result.startswith("Error"):
        return result
    
    # Handle case where result might still be the params dict
    if isinstance(result, dict) and 'action' in result:
        return f"❌ Error: Received parameters instead of data: {result}"
    
    if action == "get_total_orders":
        if isinstance(result, int):
            return f"**Total Orders for {username} in {domain} from {start_date} to {end_date}:**\n\n{result}"
        return str(result)
    elif action == "get_attendance":
        if result and isinstance(result, list):
            attendance_summary = f"**Attendance records for {username} in {domain} from {start_date} to {end_date}:**\n\n"
            for record in result:
                date = record.get('date_timezone', 'N/A')
                status = record.get('attendance_status', 'N/A')
                message = record.get('message', '')
                leave_status = record.get('leave_status', '')
                leave_type = record.get('leave_type', '')
                day = record.get('day', '')
                attendance_summary += f"• **{date}**: {status} ({day})"
                if message:
                    attendance_summary += f", _{message}_"
                if status == 'leave' and (leave_status or leave_type):
                    attendance_summary += f" [Leave: {leave_status or ''} {leave_type or ''}]"
                attendance_summary += "\n"
            return attendance_summary
        else:
            return f"No attendance records found for {username} in the specified date range."
    elif action == "get_leave_details":
        if result and isinstance(result, list):
            leave_summary = f"**Leave records for {username} in {domain} from {start_date} to {end_date}:**\n\n"
            for record in result:
                date = record.get('date_timezone', 'N/A')
                leave_status = record.get('leave_status', 'N/A')
                leave_type = record.get('leave_type', 'N/A')
                day = record.get('day', 'N/A')
                message = record.get('message', '')
                leave_summary += f"• **{date}**: {leave_status} {leave_type} ({day})"
                if message:
                    leave_summary += f", _{message}_"
                leave_summary += "\n"
            return leave_summary
        else:
            return f"No leave records found for {username} in the specified date range."
    elif action == "get_total_invoices":
        return f"📋 **Total Invoices**: {result} invoices created by {username} in {domain} from {start_date} to {end_date}"
    elif action == "calculate_sales":
        return f"💰 **Total Sales**: ${result:,.2f} in sales by {username} in {domain} from {start_date} to {end_date}"
    elif action == "get_users_and_sales_details":
        if result and isinstance(result, list) and len(result) > 0:
            sales_summary = f"**Users and Sales Details in `{domain}` from `{start_date}` to `{end_date}`:**\n\n"
            # Markdown table header
            sales_summary += "| Username | First Name | Last Name | Total Sales ($) |\n"
            sales_summary += "|---|---|---|---:|\n"
            for record in result:
                username_val = record.get('username', 'N/A')
                first_name = record.get('first_name', '')
                last_name = record.get('last_name', '')
                total_sales = record.get('total_sales', 0)
                try:
                    total_sales_float = float(total_sales)
                except (ValueError, TypeError):
                    total_sales_float = 0.0
                sales_summary += f"| {username_val} | {first_name} | {last_name} | {total_sales_float:,.2f} |\n"
            return sales_summary
        else:
            return f"No sales records found for users in `{domain}` in the specified date range."
    
    return str(result)

def check_mcp_connection():
    """Check MCP server connection and get available databases"""
    async def _check_connection():
        try:
            # Add delay to allow server to start
            await asyncio.sleep(2)
            
            async with MCPClient() as mcp_client:
                # Try to list tools as a connection test
                tools = await mcp_client.list_tools()
                print(f"DEBUG: Available tools: {[tool.name for tool in tools]}")
                
                # Try to fetch databases with timeout
                result = await asyncio.wait_for(
                    mcp_client.call_tool("list_databases", {}),
                    timeout=10.0  # 10 second timeout
                )
                return True, result
        except asyncio.TimeoutError:
            print("DEBUG: Connection check timed out")
            return False, "Connection timeout - server may not be ready"
        except Exception as e:
            print(f"DEBUG: Connection check failed: {e}")
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
    print(f"DEBUG: Connection status: {connected}, db_result: {db_result}")
except Exception as e:
    connected, db_result = False, str(e)
    print(f"DEBUG: Connection exception: {e}")

if not connected:
    st.error(f"⚠️ Cannot connect to MCP server: {db_result}")
    st.info("Make sure your MCP server is running. The server should be started as a separate Python process.")
    
    # Add information about server setup
    with st.expander("🔧 Server Setup Instructions"):
        st.code("""
            # Start the MCP server in a separate terminal:
            python server.py

            # The server runs as a stdio-based MCP server
            # The chatbot connects to it via subprocess
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
    print(f"DEBUG: parse_mcp_result input: {db_result}, type: {type(db_result)}")
    
    if isinstance(db_result, str):
        try:
            parsed = json.loads(db_result)
            print(f"DEBUG: Successfully parsed string as JSON: {parsed}")
            return parsed
        except json.JSONDecodeError as e:
            print(f"DEBUG: Failed to parse string as JSON: {e}")
            return None
    
    print(f"DEBUG: Returning db_result as-is: {db_result}")
    return db_result

if connected:
    parsed_result = parse_mcp_result(db_result)
    print(f"DEBUG: Parsed database result: {parsed_result}")
    
    if isinstance(parsed_result, list):
        try:
            db_map = {row['domain']: row['db_name'] for row in parsed_result if isinstance(row, dict) and 'domain' in row and 'db_name' in row}
            print(f"DEBUG: Created db_map: {db_map}")
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
    st.write("• Leave details for [username]")
    st.write("• Sales data for [username]")
    st.write("• Users and their sales details")
    st.write("• Date range: YYYY-MM-DD format")
    
    st.header("💡 Example Queries")
    st.code("Show total orders for antim in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30")
    st.code("Get attendance for antim in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30")
    st.code("Show leave details for antim in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30")
    st.code("Show Users and their sales details in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30")

# Chat interface
user_input = st.chat_input("Ask about your e-commerce data...")

if user_input:
    # Add user message to history
    st.session_state.history.append({"role": "user", "content": user_input})
    print(f"DEBUG: User input: {user_input}")
    
    # Fetch available actions from MCP client
    available_actions = get_available_actions()
    # Parse user intent with LLM
    prompt = build_prompt(user_input, list(db_map.keys()), available_actions)
    
    try:
        content = call_llm(prompt)
        print(f"DEBUG: LLM response: {content}")
        
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
            
            print(f"DEBUG: Cleaned LLM content: {clean_content}")
            params = json.loads(clean_content)
            print(f"DEBUG: Parsed LLM params: {params}")
            
            action = params.get("action")
            domain = params.get("domain")
            username = params.get("username")
            start_date = params.get("start_date")
            end_date = params.get("end_date")
            
            # Validate required parameters
            required_params = ["action", "domain", "start_date", "end_date"]
            if action != "get_users_and_sales_details":
                required_params.append("username")
            
            missing_params = [param for param in required_params if not params.get(param)]
            
            if missing_params:
                reply = f"❌ Missing required information: {', '.join(missing_params)}. Please provide all details."
            elif domain not in db_map:
                reply = f"❌ Unknown domain `{domain}`. Available domains: {', '.join(db_map.keys())}"
            else:
                # Call MCP tool
                db_name = db_map[domain]
                print(f"DEBUG: Using db_name: {db_name} for domain: {domain}")
                
                with st.spinner(f"Fetching {action} data..."):
                    result = call_mcp_tool_sync(action, db_name, username, start_date, end_date)
                    print(f"DEBUG: Final result from MCP: {result}")
                    reply = format_result(result, action, username, start_date, end_date, domain)
                
        except json.JSONDecodeError as e:
            print(f"DEBUG: JSON decode error: {e}")
            reply = content  # LLM couldn't parse - likely an error message
            
    except Exception as e:
        error_msg = f"❌ Error processing request: {e}"
        print(f"DEBUG: Processing error: {e}")
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
st.caption("💡 Example: 'Show total orders for antim in testrsfp.rapidor.co from 2025-01-01 to 2025-06-30'")