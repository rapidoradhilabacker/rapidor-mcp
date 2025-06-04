import asyncio
import json
import os
import logging
from typing import Any, Dict, List, Optional
import traceback
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.lowlevel.server import NotificationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Load environment variables
load_dotenv(dotenv_path='.env')

# Database configuration
INITIALIZED_DBS = os.getenv("INITIALIZED_DBS", "rapidor_master,rapidor_rsfp").split(",")
MASTER_DB_NAME = os.getenv("MASTER_DB_NAME")
MASTER_DB_USER = os.getenv("MASTER_DB_USER")
MASTER_DB_PASSWORD = os.getenv("MASTER_DB_PASSWORD")
MASTER_DB_HOST = os.getenv("MASTER_DB_HOST")
MASTER_DB_PORT = os.getenv("MASTER_DB_PORT")

# Global connections storage
DB_CONNECTIONS = {}

class DatabaseError(Exception):
    """Custom exception for database operations"""
    pass

def get_connection(db_name: str):
    """Return stored psycopg2 connection for an initialized DB with error handling"""
    if db_name not in DB_CONNECTIONS:
        raise DatabaseError(f"Database {db_name} not initialized or not allowed.")

    conn = DB_CONNECTIONS[db_name].get("connection")
    if conn is None:
        raise DatabaseError(f"No connection found for database {db_name}.")

    # Check if connection is closed and reconnect if needed
    if conn.closed:
        try:
            conn = psycopg2.connect(
                dbname=db_name,
                user=MASTER_DB_USER,
                password=MASTER_DB_PASSWORD,
                host=MASTER_DB_HOST,
                port=MASTER_DB_PORT,
                cursor_factory=RealDictCursor,
            )
            DB_CONNECTIONS[db_name]["connection"] = conn
            logger.info(f"Reconnected to database: {db_name}")
        except psycopg2.Error as e:
            raise DatabaseError(f"Database reconnection error: {e}")

    return conn

async def initialize_databases():
    """Initialize database connections for configured databases"""
    global DB_CONNECTIONS

    try:
        logger.info("Initializing database connections...")
        master_conn = psycopg2.connect(
            dbname=MASTER_DB_NAME,
            user=MASTER_DB_USER,
            password=MASTER_DB_PASSWORD,
            host=MASTER_DB_HOST,
            port=MASTER_DB_PORT,
            cursor_factory=RealDictCursor,
        )

        master_cur = master_conn.cursor()
        sql = 'SELECT "database" AS db_name, domain FROM customer_customer WHERE "database" = ANY(%s)'
        master_cur.execute(sql, (INITIALIZED_DBS,))

        for row in master_cur.fetchall():
            db_name = row["db_name"]
            domain = row["domain"]

            try:
                conn = psycopg2.connect(
                    dbname=db_name,
                    user=MASTER_DB_USER,
                    password=MASTER_DB_PASSWORD,
                    host=MASTER_DB_HOST,
                    port=MASTER_DB_PORT,
                    cursor_factory=RealDictCursor,
                    connect_timeout=60,
                )

                DB_CONNECTIONS[db_name] = {
                    "domain": domain,
                    "connection": conn
                }
                logger.info(f"Connected to database: {db_name} (domain: {domain})")

            except Exception as conn_e:
                logger.error(f"Error connecting to DB {db_name}: {conn_e}")

        master_conn.close()
        logger.info(f"Initialized {len(DB_CONNECTIONS)} database connections")

    except Exception as e:
        logger.error(f"Error initializing DB_CONNECTIONS: {e}")
        raise

# Create the server instance
server = Server("EcommerceAnalytics")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available tools"""
    return [
        Tool(
            name="list_databases",
            description="List all available initialized customer databases",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_total_orders",
            description="Get total orders count for a user within date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "username": {"type": "string", "description": "Username of the user"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "username", "start_date", "end_date"]
            }
        ),
        Tool(
            name="get_total_invoices",
            description="Get total invoices count for a user within date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "username": {"type": "string", "description": "Username of the user"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "username", "start_date", "end_date"]
            }
        ),
        Tool(
            name="get_attendance",
            description="Get attendance records for a user within date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "username": {"type": "string", "description": "Username of the user"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "username", "start_date", "end_date"]
            }
        ),
        Tool(
            name="get_leave_details",
            description="Get leave details for a user within date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "username": {"type": "string", "description": "Username of the user"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "username", "start_date", "end_date"]
            }
        ),
        Tool(
            name="calculate_sales",
            description="Calculate total sales for a user within date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "username": {"type": "string", "description": "Username of the user"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "username", "start_date", "end_date"]
            }
        ),
        Tool(
            name="get_users_and_sales_details",
            description="Get users and their sales details within a date range",
            inputSchema={
                "type": "object",
                "properties": {
                    "db_name": {"type": "string", "description": "Name of the database"},
                    "start_date": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
                    "end_date": {"type": "string", "description": "End date (YYYY-MM-DD)"}
                },
                "required": ["db_name", "start_date", "end_date"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls"""
    try:
        logger.info(f"Calling tool: {name} with arguments: {arguments}")
        
        if name == "list_databases":
            if not DB_CONNECTIONS:
                logger.warning("No databases initialized!")
                return [TextContent(type="text", text="[]")]
            result = [
                {"db_name": db_name, "domain": DB_CONNECTIONS[db_name].get("domain", "")}
                for db_name in DB_CONNECTIONS
            ]
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_total_orders":
            result = await get_total_orders_impl(
                arguments["db_name"],
                arguments["username"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "get_total_invoices":
            result = await get_total_invoices_impl(
                arguments["db_name"],
                arguments["username"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "get_attendance":
            result = await get_attendance_impl(
                arguments["db_name"],
                arguments["username"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "get_leave_details":
            result = await get_leave_details_impl(
                arguments["db_name"],
                arguments["username"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=json.dumps(result))]

        elif name == "calculate_sales":
            result = await calculate_sales_impl(
                arguments["db_name"],
                arguments["username"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=str(result))]

        elif name == "get_users_and_sales_details":
            result = await get_users_and_sales_details_impl(
                arguments["db_name"],
                arguments["start_date"],
                arguments["end_date"]
            )
            return [TextContent(type="text", text=json.dumps(result))]

        else:
            raise ValueError(f"Unknown tool: {name}")

    except Exception as e:
        logger.error(f"Error in tool {name}: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

# Tool implementation functions
async def get_total_orders_impl(db_name: str, username: str, start_date: str, end_date: str) -> int:
    """Get total orders count for a user within date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT COUNT(*) AS total_orders
            FROM order_fullorder fo
            JOIN core_basemodel cb ON fo.basemodel_ptr_id = cb.id
            JOIN auth_user au ON cb."createdBy_id" = au.id
            WHERE au.username = %s AND cb."createdOn" BETWEEN %s::timestamp AND %s::timestamp;
        '''
        cur.execute(sql, (username, start_date, end_date))
        result = cur.fetchone()
        cur.close()
        return result['total_orders'] if result else 0
    except Exception as e:
        raise DatabaseError(f"Error fetching total orders: {e}")

async def get_total_invoices_impl(db_name: str, username: str, start_date: str, end_date: str) -> int:
    """Get total invoices count for a user within date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT COUNT(*) AS total_invoices
            FROM invoice_invoice iv
            JOIN core_basemodel cb ON iv.basemodel_ptr_id = cb.id
            JOIN auth_user au ON cb."createdBy_id" = au.id
            WHERE au.username = %s AND cb."createdOn" BETWEEN %s::timestamp AND %s::timestamp;
        '''
        cur.execute(sql, (username, start_date, end_date))
        result = cur.fetchone()
        cur.close()
        return result['total_invoices'] if result else 0
    except Exception as e:
        raise DatabaseError(f"Error fetching total invoices: {e}")

async def get_attendance_impl(db_name: str, username: str, start_date: str, end_date: str) -> List[Dict]:
    """Get attendance records for a user within date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT username, owner_fullname, date_timezone, created_on, attendance_status, message, leave_status, leave_type, day
            FROM user_role_userattendance
            WHERE username = %s AND date_timezone BETWEEN %s AND %s
            ORDER BY date_timezone;
        '''
        cur.execute(sql, (username, start_date, end_date))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise DatabaseError(f"Error fetching attendance: {e}")

async def get_leave_details_impl(db_name: str, username: str, start_date: str, end_date: str) -> List[Dict]:
    """Get leave details for a user within date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT username, owner_fullname, date_timezone, created_on, attendance_status, message, leave_status, leave_type, day
            FROM user_role_userattendance
            WHERE username = %s AND date_timezone BETWEEN %s AND %s AND attendance_status = 'leave'
            ORDER BY date_timezone;
        '''
        cur.execute(sql, (username, start_date, end_date))
        columns = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        cur.close()
        return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        raise DatabaseError(f"Error fetching leave details: {e}")

async def calculate_sales_impl(db_name: str, username: str, start_date: str, end_date: str) -> float:
    """Calculate total sales for a user within date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT COALESCE(SUM(fo.order_grand_total), 0) AS total_sales
            FROM order_fullorder fo
            JOIN core_basemodel cb ON fo.basemodel_ptr_id = cb.id
            JOIN auth_user au ON cb."createdBy_id" = au.id
            WHERE au.username = %s AND cb."createdOn" BETWEEN %s::timestamp AND %s::timestamp;
        '''
        cur.execute(sql, (username, start_date, end_date))
        result = cur.fetchone()
        cur.close()
        return float(result['total_sales']) if result and result['total_sales'] else 0.0
    except Exception as e:
        raise DatabaseError(f"Error fetching total sales: {e}")

async def get_users_and_sales_details_impl(db_name: str, start_date: str, end_date: str) -> List[Dict]:
    """Get users and their sales details within a date range"""
    try:
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT au.username, au.first_name, au.last_name, SUM(fo.order_grand_total) AS total_sales
            FROM auth_user au
            JOIN core_basemodel cb ON au.id = cb."createdBy_id"
            JOIN order_fullorder fo ON cb.id = fo.basemodel_ptr_id
            WHERE cb."createdOn" BETWEEN %s::timestamp AND %s::timestamp
            GROUP BY au.username, au.first_name, au.last_name;
        '''
        cur.execute(sql, (start_date, end_date))
        rows = cur.fetchall()
        cur.close()
        transformed_result = []
        for row in rows:
            transformed_result.append({
                "username": row["username"],
                "first_name": row["first_name"],
                "last_name": row["last_name"],
                "total_sales": float(row["total_sales"]) if row["total_sales"] else 0.0
            })
        return transformed_result
    except Exception as e:
        raise DatabaseError(f"Error fetching users and sales details: {e}")

async def main():
    """Main function to run the MCP server"""
    logger.info("Starting MCP server initialization...")
    
    # Initialize database connections
    try:
        await initialize_databases()
        logger.info("Database connections initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize databases: {e}")
        return
    
    try:
        # Run the server
        logger.info("Starting stdio server...")
        async with stdio_server() as (read_stream, write_stream):
            notification_options = NotificationOptions(
                prompts_changed=True,
                resources_changed=True,
                tools_changed=True
            )
            
            logger.info("Server ready - waiting for connections...")
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="EcommerceAnalytics",
                    server_version="1.0.0",
                    capabilities=server.get_capabilities(
                        notification_options=notification_options,
                        experimental_capabilities={},
                    ),
                ),
            )
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        traceback.print_exc()
        raise
    finally:
        # Clean up database connections
        logger.info("Cleaning up database connections...")
        for db_name, db_info in DB_CONNECTIONS.items():
            try:
                if db_info.get("connection") and not db_info["connection"].closed:
                    db_info["connection"].close()
                    logger.info(f"Closed connection to {db_name}")
            except Exception as e:
                logger.error(f"Error closing connection to {db_name}: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Fatal server error: {e}")
        exit(1)