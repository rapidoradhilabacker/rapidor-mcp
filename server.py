from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from dotenv import load_dotenv
import json

load_dotenv(dotenv_path='.env')
INITIALIZED_DBS = os.getenv("INITIALIZED_DBS", "rapidor_master,rapidor_rsfp").split(",")
MASTER_DB_NAME = os.getenv("MASTER_DB_NAME")
MASTER_DB_USER = os.getenv("MASTER_DB_USER")
MASTER_DB_PASSWORD = os.getenv("MASTER_DB_PASSWORD")
MASTER_DB_HOST = os.getenv("MASTER_DB_HOST")
MASTER_DB_PORT = os.getenv("MASTER_DB_PORT")
# Initialize FastMCP server
mcp = FastMCP("EcommerceAnalytics")

def get_connection(db_name):
    """Create database connection for an initialized DB with error handling"""
    if db_name not in DB_CONNECTIONS:
        raise ToolError(f"Database {db_name} not initialized or not allowed.")
    params = DB_CONNECTIONS[db_name]
    try:
        return psycopg2.connect(
            dbname=db_name,
            user=MASTER_DB_USER,
            password=MASTER_DB_PASSWORD,
            host=MASTER_DB_HOST,
            port=MASTER_DB_PORT,
            cursor_factory=RealDictCursor,
        )
    except psycopg2.Error as e:
        raise ToolError(f"Database connection error: {e}")

@mcp.tool()
def list_databases() -> list[dict]:
    """List all available initialized customer databases"""
    # Return the list of initialized DBs and their domains
    return [
        {"db_name": db_name, "domain": DB_CONNECTIONS[db_name]["domain"]}
        for db_name in DB_CONNECTIONS
    ]

@mcp.tool()
def get_total_orders(
    db_name: str,
    username: str,
    start_date: str,
    end_date: str,
) -> int:
    """Get total orders count for a user within date range (initialized DBs only)"""
    try:
        print("Starting get_total_orders")
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
        conn.close()
        return result['total_orders'] if result else 0
    except Exception as e:
        raise ToolError(f"Error fetching total orders: {e}")

@mcp.tool()
def get_total_invoices(
    db_name: str,
    username: str,
    start_date: str,
    end_date: str,
) -> int:
    """Get total invoices count for a user within date range (initialized DBs only)"""
    try:
        print("Starting get_total_invoices")
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
        conn.close()
        return result['total_invoices'] if result else 0
    except Exception as e:
        raise ToolError(f"Error fetching total invoices: {e}")

@mcp.tool()
def get_attendance(
    db_name: str,
    username: str,
    start_date: str,
    end_date: str,
) -> list[dict]:
    """Get attendance records for a user within date range (initialized DBs only)"""
    try:
        print("Starting get_attendance")
        conn = get_connection(db_name)
        cur = conn.cursor()
        sql = '''
            SELECT at.date AS attendance_date, at.status, at.check_in, at.check_out
            FROM attendance_attendancelog at
            JOIN auth_user au ON at.user_id = au.id
            WHERE au.username = %s AND at.date BETWEEN %s AND %s
            ORDER BY at.date;
        '''
        cur.execute(sql, (username, start_date, end_date))
        rows = cur.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        raise ToolError(f"Error fetching attendance: {e}")

@mcp.tool()
def calculate_sales(
    db_name: str,
    username: str,
    start_date: str,
    end_date: str,
) -> float:
    """Calculate total sales for a user within date range (initialized DBs only)"""
    try:
        print("Starting calculate_sales")
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
        conn.close()
        return float(result['total_sales']) if result and result['total_sales'] else 0.0
    except Exception as e:
        raise ToolError(f"Error fetching total sales: {e}")

# --- Initialize DB_CONNECTIONS for INITIALIZED_DBS ---
DB_CONNECTIONS = {}
try:
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
        DB_CONNECTIONS[row["db_name"]] = {
            "domain": row["domain"]
        }
    master_conn.close()
except Exception as e:
    print(f"Error initializing DB_CONNECTIONS: {e}")

if __name__ == "__main__":
    mcp.run(transport="sse", host="127.0.0.1", port=8087)
