import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Literal, Optional, Union
from sqlalchemy import create_engine, Engine
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits

# Load OPENAI_API_KEY (and optional OPENAI_MODEL override) from a local .env file.
dotenv.load_dotenv()
os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

_raw_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
MODEL = _raw_model if ":" in _raw_model else f"openai:{_raw_model}"


DETERMINISTIC_SETTINGS = {"temperature": 0.0}

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

# per-item retail price
UNIT_PRICE_BY_ITEM: Dict[str, float] = {item["item_name"]: item["unit_price"] for item in paper_supplies}
CANONICAL_ITEM_NAMES = tuple(UNIT_PRICE_BY_ITEM.keys())

ItemName = Literal[CANONICAL_ITEM_NAMES]


class LineItem(BaseModel):
    """A single requested paper product, resolved to a canonical catalog name."""

    item_name: ItemName
    quantity: int = Field(gt=0)


def normalize_iso_date(date_str: Optional[str]) -> Optional[str]:
    """Best-effort coercion of a model-provided date string to YYYY-MM-DD, else None."""
    if not date_str:
        return None
    try:
        return datetime.fromisoformat(date_str[:10]).strftime("%Y-%m-%d")
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Inventory Agent: checks stock, decides on and places reorders.
# ----------------------------------------------------------------------------


class ItemAvailability(BaseModel):
    item_name: ItemName
    requested_quantity: int
    fulfillable: bool
    available_by: str  
    reorder_placed: bool
    note: str


class InventoryAssessment(BaseModel):
    items: List[ItemAvailability]
    all_fulfillable: bool


inventory_agent = Agent(
    MODEL,
    output_type=InventoryAssessment,
    model_settings=DETERMINISTIC_SETTINGS,
    system_prompt=(
        "You are the Inventory Agent for Beaver's Choice Paper Company. "
        "You are given requested items (item_name, quantity) and the request date. "
        "For EVERY item: "
        "1) Call check_stock to get current stock. "
        "2) If current stock >= requested quantity, the item is fulfillable and "
        "available_by equals the request date, reorder_placed=False. "
        "3) If stock is insufficient, call estimate_restock_delivery with the shortfall "
        "(requested minus current stock) to learn the earliest delivery date, and call "
        "check_cash to confirm the shortfall purchase (shortfall * unit price) does not "
        "exceed the available cash balance. "
        "4) If affordable, call reorder_stock with the shortfall quantity, then mark the "
        "item fulfillable=True, reorder_placed=True, and available_by = the date "
        "estimate_restock_delivery returned. "
        "5) If NOT affordable, mark the item fulfillable=False, reorder_placed=False, and "
        "explain why in note. "
        "Never invent numbers you did not get from a tool. Keep each note under 20 words."
    ),
)


@inventory_agent.tool_plain
def check_stock(item_name: ItemName, as_of_date: str) -> int:
    """Return current on-hand stock for an item as of a date. Wraps get_stock_level."""
    stock_df = get_stock_level(item_name, as_of_date)
    return int(stock_df["current_stock"].iloc[0])


@inventory_agent.tool_plain
def estimate_restock_delivery(as_of_date: str, quantity: int) -> str:
    """Estimate the supplier delivery date for a reorder. Wraps get_supplier_delivery_date."""
    return get_supplier_delivery_date(as_of_date, quantity)


@inventory_agent.tool_plain
def check_cash(as_of_date: str) -> float:
    """Return the company's current cash balance. Wraps get_cash_balance."""
    return get_cash_balance(as_of_date)


@inventory_agent.tool_plain
def reorder_stock(item_name: ItemName, quantity: int, as_of_date: str) -> int:
    """Record a supplier stock order for an item. Wraps create_transaction(stock_orders)."""
    cost = round(quantity * UNIT_PRICE_BY_ITEM[item_name], 2)
    return create_transaction(item_name, "stock_orders", quantity, cost, as_of_date)


async def call_inventory_agent(line_items: List[LineItem], as_of_date: str) -> InventoryAssessment:
    prompt = (
        f"Request date: {as_of_date}\n"
        f"Requested items: {[li.model_dump() for li in line_items]}\n"
        "Assess availability for every item."
    )
    result = await inventory_agent.run(prompt)
    return result.output


# ----------------------------------------------------------------------------
# Quoting Agent: prices the order and explains the discount to the customer.
# ----------------------------------------------------------------------------


def price_line_items(line_items: List[LineItem]) -> float:
    return round(sum(UNIT_PRICE_BY_ITEM[li.item_name] * li.quantity for li in line_items), 2)


def discount_tier(total_quantity: int) -> float:
    """Deterministic bulk-discount schedule, kept outside the LLM so pricing is reproducible."""
    if total_quantity >= 1000:
        return 0.15
    if total_quantity >= 500:
        return 0.10
    if total_quantity >= 100:
        return 0.05
    return 0.0


class QuoteResult(BaseModel):
    subtotal: float
    discount_pct: float
    total_price: float
    explanation: str


quoting_agent = Agent(
    MODEL,
    output_type=QuoteResult,
    model_settings=DETERMINISTIC_SETTINGS,
    system_prompt=(
        "You are the Quoting Agent for Beaver's Choice Paper Company. "
        "You will be given the itemized subtotal and the bulk-discount percentage that has "
        "already been calculated for this order, plus its context. "
        "Call find_similar_quotes with a few relevant keywords (item names, job type, or "
        "event type) to see how similar past orders were described, then write a short, "
        "friendly, customer-facing explanation of the price that itemizes what was ordered "
        "and states the discount applied and why (e.g. order size). "
        "Never mention internal costs, margins, or supplier prices. "
        "Return subtotal, discount_pct, and total_price exactly as given to you -- do not "
        "recompute them."
    ),
)


@quoting_agent.tool_plain
def find_similar_quotes(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """Look up similar historical quotes for pricing/discount context. Wraps search_quote_history."""
    return search_quote_history(search_terms, limit)


async def build_quote(line_items: List[LineItem], customer_context: str) -> QuoteResult:
    """Deterministically price the order, then have the Quoting Agent explain it."""
    subtotal = price_line_items(line_items)
    total_quantity = sum(li.quantity for li in line_items)
    pct = discount_tier(total_quantity)
    total_price = round(subtotal * (1 - pct), 2)

    prompt = (
        f"Order context: {customer_context}\n"
        f"Items: {[li.model_dump() for li in line_items]}\n"
        f"Subtotal: ${subtotal:.2f}\n"
        f"Discount already decided: {pct:.0%}\n"
        f"Total price already decided: ${total_price:.2f}\n"
        "Write the customer-facing explanation now."
    )
    result = await quoting_agent.run(prompt)
    quote = result.output
    # The numbers are deterministic business logic; don't let the model drift from them.
    quote.subtotal, quote.discount_pct, quote.total_price = subtotal, pct, total_price
    return quote


# ----------------------------------------------------------------------------
# Sales Agent: final feasibility gate -- finalize the sale or reject the order.
# ----------------------------------------------------------------------------


class SalesDecision(BaseModel):
    fulfilled: bool
    transaction_ids: List[int]
    rejection_reason: Optional[str]


sales_agent = Agent(
    MODEL,
    output_type=SalesDecision,
    model_settings=DETERMINISTIC_SETTINGS,
    system_prompt=(
        "You are the Sales Agent for Beaver's Choice Paper Company, the final gate before "
        "an order is confirmed. You are given the requested items, the inventory "
        "availability assessment, the per-item price after discount, and whether every item "
        "can be delivered in time for the customer's requested date (if any). "
        "Only finalize the order if ALL items are fulfillable AND available in time. In that "
        "case, call run_financial_health_check once as a sanity check, then call "
        "finalize_sale once per item with its price, and return fulfilled=True with the "
        "resulting transaction ids. "
        "If ANY item fails either condition, reject the WHOLE order: fulfilled=False, "
        "transaction_ids=[], and a specific, customer-safe rejection_reason that names the "
        "exact item(s) at fault and the exact cause -- insufficient stock, supplier lead "
        "time longer than the requested delivery date, or inability to afford a reorder. "
        "Never use vague language like 'not fully confirmed'; always state which item and "
        "why. Never reveal internal cost, margin, or cash-balance figures in the "
        "rejection_reason."
    ),
)


@sales_agent.tool_plain
def run_financial_health_check(as_of_date: str) -> Dict:
    """Snapshot cash/inventory position before committing to the order. Wraps generate_financial_report."""
    return generate_financial_report(as_of_date)


@sales_agent.tool_plain
def finalize_sale(item_name: ItemName, quantity: int, price: float, as_of_date: str) -> int:
    """Record a finalized sale. Wraps create_transaction(sales)."""
    return create_transaction(item_name, "sales", quantity, price, as_of_date)


async def call_sales_agent(
    line_items: List[LineItem],
    availability: InventoryAssessment,
    quote: QuoteResult,
    requested_delivery_date: Optional[str],
    as_of_date: str,
) -> SalesDecision:
    # Delivery-date feasibility is simple date comparison -- compute it in Python rather
    # than trusting the LLM to get 70 date comparisons right.
    on_time = True
    if requested_delivery_date:
        on_time = all(item.available_by <= requested_delivery_date for item in availability.items)

    per_item_price = {
        li.item_name: round(UNIT_PRICE_BY_ITEM[li.item_name] * li.quantity * (1 - quote.discount_pct), 2)
        for li in line_items
    }
    prompt = (
        f"Request date: {as_of_date}\n"
        f"Customer requested delivery by: {requested_delivery_date or 'no specific date given'}\n"
        f"Items requested: {[li.model_dump() for li in line_items]}\n"
        f"Inventory availability (all_fulfillable={availability.all_fulfillable}): "
        f"{[item.model_dump() for item in availability.items]}\n"
        f"All items available in time for the requested delivery date: {on_time}\n"
        f"Per-item price after discount: {per_item_price}\n"
        f"Order total: ${quote.total_price:.2f}\n"
        "Decide whether to finalize or reject this whole order."
    )
    result = await sales_agent.run(prompt)
    return result.output


# ----------------------------------------------------------------------------
# Orchestrator Agent: parses the request, delegates in a fixed sequence
# (Inventory -> Quoting -> Sales), and composes the customer-facing reply.
# ----------------------------------------------------------------------------


class OrderOutcome(BaseModel):
    fulfilled: bool
    total_price: Optional[float] = None
    quote_explanation: Optional[str] = None
    transaction_ids: List[int] = Field(default_factory=list)
    rejection_reason: Optional[str] = None


class CustomerResponse(BaseModel):
    fulfilled: bool
    message: str


orchestrator_agent = Agent(
    MODEL,
    output_type=CustomerResponse,
    model_settings=DETERMINISTIC_SETTINGS,
    system_prompt=(
        "You are the customer-facing Orchestrator for Beaver's Choice Paper Company. "
        "A customer sent a free-text request for paper supplies, possibly mentioning a "
        "delivery deadline. "
        "1) Identify every distinct item requested. Match each one to the single closest "
        "item in this exact catalog, by its core product noun (e.g. 'cardstock' in the "
        "request matches catalog item 'Cardstock', not 'Heavyweight paper'; prefer a match "
        "on the material/product name over incidental adjectives like 'heavy' or 'quality'). "
        f"Catalog: {', '.join(CANONICAL_ITEM_NAMES)}. "
        "If a requested item has no reasonably close catalog match (e.g. a size or product "
        "we clearly don't stock), DO NOT force a mapping -- leave it out of line_items "
        "entirely, and mention in your final message that we don't carry that specific item. "
        "With its quantity, note any explicit delivery deadline mentioned, formatted as "
        "YYYY-MM-DD, or null if none is stated. Write a one-sentence customer_context "
        "describing the job/event/order size. "
        "2) If at least one item resolved to a catalog match, call process_order with those "
        "items, customer_context, requested_delivery_date, and the request date. If NO "
        "items resolved, skip process_order and compose a rejection message explaining none "
        "of the requested items are in our catalog. "
        "3) Compose the final response: fulfilled must match process_order's fulfilled value "
        "(or False if you skipped it). If fulfilled, write a friendly message stating what "
        "was ordered and the total price, including the tool's quote_explanation, and "
        "mention any items you left out as not carried. If not fulfilled, write a friendly "
        "message giving the rejection reason. Never include internal system details (cash "
        "balance, margins, stock counts, raw errors)."
    ),
)


@orchestrator_agent.tool_plain
async def process_order(
    line_items: List[LineItem],
    customer_context: str,
    requested_delivery_date: Optional[str],
    as_of_date: str,
) -> OrderOutcome:
    """Delegate the parsed order to the Inventory, Quoting, and Sales agents in sequence."""
    availability = await call_inventory_agent(line_items, as_of_date)
    quote = await build_quote(line_items, customer_context)
    decision = await call_sales_agent(
        line_items, availability, quote, normalize_iso_date(requested_delivery_date), as_of_date
    )
    return OrderOutcome(
        fulfilled=decision.fulfilled,
        total_price=quote.total_price if decision.fulfilled else None,
        quote_explanation=quote.explanation if decision.fulfilled else None,
        transaction_ids=decision.transaction_ids,
        rejection_reason=decision.rejection_reason,
    )


def handle_customer_request(request_text: str, as_of_date: str) -> CustomerResponse:
    """Run one customer request through the full multi-agent pipeline."""
    prompt = f"Request date: {as_of_date}\nCustomer request:\n{request_text}"
    result = orchestrator_agent.run_sync(prompt, usage_limits=UsageLimits(request_limit=200))
    return result.output


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    # The multi-agent system is module-level (inventory_agent, quoting_agent, sales_agent,
    # orchestrator_agent) and ready to use via handle_customer_request().

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        try:
            agent_response = handle_customer_request(request_with_date, request_date)
            fulfilled = agent_response.fulfilled
            response = agent_response.message
        except Exception as e:
            fulfilled = False
            response = "We're sorry, we were unable to process this request due to a system error."
            print(f"ERROR processing request {idx + 1}: {e}")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "fulfilled": fulfilled,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
