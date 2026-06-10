"""
Build data/sample.sqlite and generate NL-to-SQL dataset.
Every gold SQL is executed against the DB before being written; nothing fabricated.

Usage:
    python data/build_dataset.py

Outputs:
    data/sample.sqlite      — retail database
    data/train.jsonl        — 160 validated (question, schema, sql) triples
    data/heldout.jsonl      —  40 validated triples (held out for eval)
"""

import json
import random
import sqlite3
from pathlib import Path

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).parent / "sample.sqlite"
TRAIN_PATH = Path(__file__).parent / "train.jsonl"
HELDOUT_PATH = Path(__file__).parent / "heldout.jsonl"

SCHEMA_DDL = """
CREATE TABLE products (
    product_id   INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    category     TEXT    NOT NULL,
    price        REAL    NOT NULL,
    stock_qty    INTEGER NOT NULL
);

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    name         TEXT    NOT NULL,
    email        TEXT    NOT NULL UNIQUE,
    city         TEXT    NOT NULL,
    country      TEXT    NOT NULL,
    joined_date  TEXT    NOT NULL   -- ISO-8601 YYYY-MM-DD
);

CREATE TABLE orders (
    order_id     INTEGER PRIMARY KEY,
    customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
    order_date   TEXT    NOT NULL,  -- YYYY-MM-DD
    status       TEXT    NOT NULL   -- pending / shipped / delivered / cancelled
);

CREATE TABLE order_items (
    item_id      INTEGER PRIMARY KEY,
    order_id     INTEGER NOT NULL REFERENCES orders(order_id),
    product_id   INTEGER NOT NULL REFERENCES products(product_id),
    quantity     INTEGER NOT NULL,
    unit_price   REAL    NOT NULL
);
"""

SCHEMA_TEXT = (
    "products(product_id, name, category, price, stock_qty) | "
    "customers(customer_id, name, email, city, country, joined_date) | "
    "orders(order_id, customer_id, order_date, status) | "
    "order_items(item_id, order_id, product_id, quantity, unit_price)"
)


# --------------------------------------------------------------------------- #
#  Sample data                                                                  #
# --------------------------------------------------------------------------- #

PRODUCTS = [
    (1,  "Laptop Pro 15",       "Electronics",  1299.99, 42),
    (2,  "Wireless Mouse",      "Electronics",    29.99, 230),
    (3,  "USB-C Hub 7-port",    "Electronics",    49.99, 180),
    (4,  "Mechanical Keyboard", "Electronics",    89.99,  95),
    (5,  "27-inch Monitor",     "Electronics",   349.99,  60),
    (6,  "Noise-Cancel Headphones", "Electronics", 199.99, 75),
    (7,  "Webcam 4K",           "Electronics",    79.99, 120),
    (8,  "Standing Desk",       "Furniture",     499.99,  30),
    (9,  "Ergonomic Chair",     "Furniture",     379.99,  25),
    (10, "Desk Lamp LED",       "Furniture",      39.99, 160),
    (11, "Notebook A5 Pack",    "Stationery",      9.99, 500),
    (12, "Ballpoint Pens 10pk", "Stationery",      4.99, 800),
    (13, "Sticky Notes 6pk",    "Stationery",      6.99, 600),
    (14, "Whiteboard 60x90",    "Stationery",     59.99,  55),
    (15, "Coffee Machine Pro",  "Appliances",    149.99,  40),
    (16, "Electric Kettle 1.7L","Appliances",     34.99, 110),
    (17, "Air Purifier",        "Appliances",    229.99,  28),
    (18, "Smart Plug 4-pack",   "Electronics",    24.99, 200),
    (19, "Cable Management Box","Accessories",    19.99, 300),
    (20, "Monitor Arm Dual",    "Accessories",    89.99,  70),
]

CITIES = ["Berlin", "Paris", "London", "Madrid", "Rome", "Amsterdam", "Vienna", "Warsaw"]
COUNTRIES = {"Berlin": "Germany", "Paris": "France", "London": "UK",
             "Madrid": "Spain", "Rome": "Italy", "Amsterdam": "Netherlands",
             "Vienna": "Austria", "Warsaw": "Poland"}
FIRST = ["Alice", "Bob", "Carol", "David", "Emma", "Frank", "Grace", "Henry",
         "Isabella", "James", "Karen", "Liam", "Mia", "Noah", "Olivia", "Paul"]
LAST  = ["Smith", "Mueller", "Dupont", "Garcia", "Rossi", "Brown", "Janssen",
         "Schmidt", "Martin", "Wilson"]

rng = random.Random(SEED)
CUSTOMERS = []
for i in range(1, 101):
    fn = rng.choice(FIRST)
    ln = rng.choice(LAST)
    city = rng.choice(CITIES)
    year = rng.randint(2019, 2023)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    CUSTOMERS.append((
        i, f"{fn} {ln}",
        f"{fn.lower()}.{ln.lower()}{i}@example.com",
        city, COUNTRIES[city],
        f"{year}-{month:02d}-{day:02d}",
    ))

STATUSES = ["pending", "shipped", "delivered", "cancelled"]
ORDERS = []
ORDER_ITEMS = []
item_counter = 1
for order_id in range(1, 201):
    cust_id = rng.randint(1, 100)
    year = rng.randint(2022, 2024)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    status = rng.choices(STATUSES, weights=[10, 20, 60, 10])[0]
    ORDERS.append((order_id, cust_id, f"{year}-{month:02d}-{day:02d}", status))
    n_items = rng.randint(1, 5)
    chosen = rng.sample(range(1, 21), n_items)
    for pid in chosen:
        qty = rng.randint(1, 4)
        price = next(p[3] for p in PRODUCTS if p[0] == pid)
        ORDER_ITEMS.append((item_counter, order_id, pid, qty, price))
        item_counter += 1


# --------------------------------------------------------------------------- #
#  Build DB                                                                     #
# --------------------------------------------------------------------------- #

def build_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_DDL)
    conn.executemany("INSERT INTO products VALUES (?,?,?,?,?)", PRODUCTS)
    conn.executemany("INSERT INTO customers VALUES (?,?,?,?,?,?)", CUSTOMERS)
    conn.executemany("INSERT INTO orders VALUES (?,?,?,?)", ORDERS)
    conn.executemany("INSERT INTO order_items VALUES (?,?,?,?,?)", ORDER_ITEMS)
    conn.commit()
    conn.close()
    print(f"DB built: {DB_PATH} — "
          f"{len(PRODUCTS)} products, {len(CUSTOMERS)} customers, "
          f"{len(ORDERS)} orders, {len(ORDER_ITEMS)} items")


# --------------------------------------------------------------------------- #
#  NL→SQL pairs                                                                 #
# --------------------------------------------------------------------------- #

PAIRS = [
    # ---- basic SELECT ----
    ("List all product names and their prices.",
     "SELECT name, price FROM products ORDER BY name;"),
    ("Show all product categories without duplicates.",
     "SELECT DISTINCT category FROM products ORDER BY category;"),
    ("How many products are in the database?",
     "SELECT COUNT(*) AS total_products FROM products;"),
    ("Show the 5 most expensive products.",
     "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"),
    ("Which products cost less than $30?",
     "SELECT name, price FROM products WHERE price < 30 ORDER BY price;"),
    ("Show products with more than 100 items in stock.",
     "SELECT name, stock_qty FROM products WHERE stock_qty > 100 ORDER BY stock_qty DESC;"),
    ("What is the average product price?",
     "SELECT ROUND(AVG(price), 2) AS avg_price FROM products;"),
    ("What is the total value of all inventory?",
     "SELECT ROUND(SUM(price * stock_qty), 2) AS inventory_value FROM products;"),
    ("Find the cheapest product.",
     "SELECT name, price FROM products ORDER BY price LIMIT 1;"),
    ("Which product has the highest stock quantity?",
     "SELECT name, stock_qty FROM products ORDER BY stock_qty DESC LIMIT 1;"),

    # ---- category aggregations ----
    ("How many products are in each category?",
     "SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY category;"),
    ("What is the average price per category?",
     "SELECT category, ROUND(AVG(price), 2) AS avg_price FROM products GROUP BY category ORDER BY category;"),
    ("Which category has the most products?",
     "SELECT category, COUNT(*) AS n FROM products GROUP BY category ORDER BY n DESC LIMIT 1;"),
    ("Show total inventory value per category.",
     "SELECT category, ROUND(SUM(price * stock_qty), 2) AS total_value FROM products "
     "GROUP BY category ORDER BY total_value DESC;"),
    ("Which categories have more than 3 products?",
     "SELECT category, COUNT(*) AS n FROM products GROUP BY category HAVING n > 3 ORDER BY n DESC;"),

    # ---- customers ----
    ("How many customers are there?",
     "SELECT COUNT(*) AS total_customers FROM customers;"),
    ("List all unique countries customers are from.",
     "SELECT DISTINCT country FROM customers ORDER BY country;"),
    ("How many customers are from Germany?",
     "SELECT COUNT(*) AS n FROM customers WHERE country = 'Germany';"),
    ("Show customers who joined in 2022.",
     "SELECT name, email, joined_date FROM customers "
     "WHERE joined_date LIKE '2022%' ORDER BY joined_date;"),
    ("How many customers are in each city?",
     "SELECT city, COUNT(*) AS n FROM customers GROUP BY city ORDER BY n DESC;"),
    ("Which city has the most customers?",
     "SELECT city, COUNT(*) AS n FROM customers GROUP BY city ORDER BY n DESC LIMIT 1;"),
    ("List customers whose names start with 'Alice'.",
     "SELECT name, email, city FROM customers WHERE name LIKE 'Alice%' ORDER BY name;"),
    ("How many customers joined each year?",
     "SELECT strftime('%Y', joined_date) AS year, COUNT(*) AS n "
     "FROM customers GROUP BY year ORDER BY year;"),
    ("Show customers from France or Germany.",
     "SELECT name, city, country FROM customers "
     "WHERE country IN ('France', 'Germany') ORDER BY country, name;"),
    ("Find customers who joined before 2021.",
     "SELECT name, joined_date FROM customers "
     "WHERE joined_date < '2021-01-01' ORDER BY joined_date;"),

    # ---- orders ----
    ("How many orders are there in total?",
     "SELECT COUNT(*) AS total_orders FROM orders;"),
    ("How many orders have each status?",
     "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY status;"),
    ("Show orders placed in 2024.",
     "SELECT order_id, customer_id, order_date, status FROM orders "
     "WHERE order_date LIKE '2024%' ORDER BY order_date;"),
    ("How many orders were cancelled?",
     "SELECT COUNT(*) AS cancelled FROM orders WHERE status = 'cancelled';"),
    ("How many orders were delivered?",
     "SELECT COUNT(*) AS delivered FROM orders WHERE status = 'delivered';"),
    ("List the 10 most recent orders.",
     "SELECT order_id, customer_id, order_date, status FROM orders "
     "ORDER BY order_date DESC LIMIT 10;"),
    ("How many orders were placed each year?",
     "SELECT strftime('%Y', order_date) AS year, COUNT(*) AS n "
     "FROM orders GROUP BY year ORDER BY year;"),
    ("Show all orders placed in January 2023.",
     "SELECT order_id, order_date, status FROM orders "
     "WHERE order_date LIKE '2023-01%' ORDER BY order_date;"),
    ("Which customers have placed more than 3 orders?",
     "SELECT customer_id, COUNT(*) AS n FROM orders "
     "GROUP BY customer_id HAVING n > 3 ORDER BY n DESC;"),
    ("What percentage of orders are delivered?",
     "SELECT ROUND(100.0 * SUM(CASE WHEN status='delivered' THEN 1 ELSE 0 END) "
     "/ COUNT(*), 1) AS pct_delivered FROM orders;"),

    # ---- order_items ----
    ("What is the total quantity sold across all orders?",
     "SELECT SUM(quantity) AS total_qty FROM order_items;"),
    ("What is the total revenue from all orders?",
     "SELECT ROUND(SUM(quantity * unit_price), 2) AS total_revenue FROM order_items;"),
    ("How many line items are in the database?",
     "SELECT COUNT(*) AS total_items FROM order_items;"),
    ("What is the average order value?",
     "SELECT ROUND(AVG(order_total), 2) AS avg_order_value FROM "
     "(SELECT order_id, SUM(quantity * unit_price) AS order_total FROM order_items "
     " GROUP BY order_id);"),

    # ---- JOINs ----
    ("Show each order with the customer name.",
     "SELECT o.order_id, c.name, o.order_date, o.status "
     "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
     "ORDER BY o.order_date DESC LIMIT 20;"),
    ("List all products ordered at least once.",
     "SELECT DISTINCT p.name, p.category FROM products p "
     "JOIN order_items oi ON p.product_id = oi.product_id ORDER BY p.name;"),
    ("Show the top 5 best-selling products by quantity.",
     "SELECT p.name, SUM(oi.quantity) AS qty_sold "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.product_id ORDER BY qty_sold DESC LIMIT 5;"),
    ("Which product generated the most revenue?",
     "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.product_id ORDER BY revenue DESC LIMIT 1;"),
    ("Show the total amount spent by each customer.",
     "SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 10;"),
    ("List customers who have never placed an order.",
     "SELECT c.name, c.email FROM customers c "
     "WHERE c.customer_id NOT IN (SELECT DISTINCT customer_id FROM orders) "
     "ORDER BY c.name;"),
    ("Which customers placed orders in 2024?",
     "SELECT DISTINCT c.name, c.city FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "WHERE o.order_date LIKE '2024%' ORDER BY c.name;"),
    ("Show products never ordered.",
     "SELECT p.name, p.category FROM products p "
     "WHERE p.product_id NOT IN (SELECT DISTINCT product_id FROM order_items) "
     "ORDER BY p.name;"),
    ("What is the total revenue per product category?",
     "SELECT p.category, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.category ORDER BY revenue DESC;"),
    ("Show the number of distinct products ordered per order.",
     "SELECT order_id, COUNT(DISTINCT product_id) AS distinct_products "
     "FROM order_items GROUP BY order_id ORDER BY distinct_products DESC LIMIT 10;"),

    # ---- subqueries / advanced ----
    ("Find products more expensive than the average price.",
     "SELECT name, price FROM products "
     "WHERE price > (SELECT AVG(price) FROM products) ORDER BY price DESC;"),
    ("Show the most recent order for each customer.",
     "SELECT customer_id, MAX(order_date) AS latest_order "
     "FROM orders GROUP BY customer_id ORDER BY latest_order DESC LIMIT 10;"),
    ("Which customers have placed exactly one order?",
     "SELECT customer_id, COUNT(*) AS n FROM orders "
     "GROUP BY customer_id HAVING n = 1 ORDER BY customer_id;"),
    ("Show the top 3 cities by total customer spend.",
     "SELECT c.city, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY c.city ORDER BY total_spent DESC LIMIT 3;"),
    ("List customers who ordered Electronics products.",
     "SELECT DISTINCT c.name, c.city "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN order_items oi ON o.order_id = oi.order_id "
     "JOIN products p ON oi.product_id = p.product_id "
     "WHERE p.category = 'Electronics' ORDER BY c.name;"),
    ("What is the average number of items per order?",
     "SELECT ROUND(AVG(item_count), 2) AS avg_items_per_order FROM "
     "(SELECT order_id, COUNT(*) AS item_count FROM order_items GROUP BY order_id);"),
    ("Show orders with a total value above $500.",
     "SELECT oi.order_id, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS order_total "
     "FROM order_items oi GROUP BY oi.order_id HAVING order_total > 500 "
     "ORDER BY order_total DESC;"),
    ("Which products from the Furniture category are low in stock (less than 30)?",
     "SELECT name, stock_qty FROM products "
     "WHERE category = 'Furniture' AND stock_qty < 30 ORDER BY stock_qty;"),
    ("Find the month with the highest number of orders.",
     "SELECT strftime('%Y-%m', order_date) AS month, COUNT(*) AS n "
     "FROM orders GROUP BY month ORDER BY n DESC LIMIT 1;"),
    ("How many orders were placed per month in 2023?",
     "SELECT strftime('%m', order_date) AS month, COUNT(*) AS n "
     "FROM orders WHERE order_date LIKE '2023%' "
     "GROUP BY month ORDER BY month;"),

    # ---- more aggregations ----
    ("What is the min and max product price in each category?",
     "SELECT category, MIN(price) AS min_price, MAX(price) AS max_price "
     "FROM products GROUP BY category ORDER BY category;"),
    ("What is the total stock across all products?",
     "SELECT SUM(stock_qty) AS total_stock FROM products;"),
    ("Show the 5 customers with the most orders.",
     "SELECT customer_id, COUNT(*) AS n FROM orders "
     "GROUP BY customer_id ORDER BY n DESC LIMIT 5;"),
    ("Which country has the most customers?",
     "SELECT country, COUNT(*) AS n FROM customers "
     "GROUP BY country ORDER BY n DESC LIMIT 1;"),
    ("What is the total revenue per year?",
     "SELECT strftime('%Y', o.order_date) AS year, "
     "ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM orders o JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY year ORDER BY year;"),
    ("Show all orders made by customer ID 1.",
     "SELECT order_id, order_date, status FROM orders "
     "WHERE customer_id = 1 ORDER BY order_date;"),
    ("How many unique customers have placed at least one order?",
     "SELECT COUNT(DISTINCT customer_id) AS ordering_customers FROM orders;"),
    ("What fraction of customers have placed an order?",
     "SELECT ROUND(100.0 * COUNT(DISTINCT o.customer_id) / "
     "(SELECT COUNT(*) FROM customers), 1) AS pct_ordered "
     "FROM orders o;"),
    ("List all Electronics products ordered by price descending.",
     "SELECT name, price FROM products "
     "WHERE category = 'Electronics' ORDER BY price DESC;"),
    ("Find orders that contain more than 3 distinct products.",
     "SELECT order_id, COUNT(DISTINCT product_id) AS n "
     "FROM order_items GROUP BY order_id HAVING n > 3 ORDER BY n DESC;"),

    # ---- conditional / CASE ----
    ("Label each product as 'cheap' (<$30), 'mid' ($30-$100), or 'expensive' (>$100).",
     "SELECT name, price, "
     "CASE WHEN price < 30 THEN 'cheap' "
     "     WHEN price <= 100 THEN 'mid' "
     "     ELSE 'expensive' END AS price_tier "
     "FROM products ORDER BY price;"),
    ("Show the count of products in each price tier (cheap/mid/expensive).",
     "SELECT "
     "SUM(CASE WHEN price < 30 THEN 1 ELSE 0 END) AS cheap, "
     "SUM(CASE WHEN price BETWEEN 30 AND 100 THEN 1 ELSE 0 END) AS mid, "
     "SUM(CASE WHEN price > 100 THEN 1 ELSE 0 END) AS expensive "
     "FROM products;"),
    ("How many orders were shipped or delivered?",
     "SELECT COUNT(*) AS n FROM orders "
     "WHERE status IN ('shipped', 'delivered');"),
    ("Show total revenue from delivered orders only.",
     "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
     "WHERE o.status = 'delivered';"),
    ("List orders placed after 2023-06-01 that are still pending.",
     "SELECT order_id, customer_id, order_date FROM orders "
     "WHERE order_date > '2023-06-01' AND status = 'pending' ORDER BY order_date;"),

    # ---- more joins ----
    ("Show customer name and total number of orders for customers from the UK.",
     "SELECT c.name, COUNT(o.order_id) AS n_orders "
     "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
     "WHERE c.country = 'UK' GROUP BY c.customer_id ORDER BY n_orders DESC;"),
    ("Which product was ordered the most times (by order count, not quantity)?",
     "SELECT p.name, COUNT(DISTINCT oi.order_id) AS n_orders "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.product_id ORDER BY n_orders DESC LIMIT 1;"),
    ("Show the average spend per order for each country.",
     "SELECT c.country, "
     "ROUND(AVG(order_total), 2) AS avg_order_spend "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN (SELECT order_id, SUM(quantity * unit_price) AS order_total "
     "      FROM order_items GROUP BY order_id) sub ON o.order_id = sub.order_id "
     "GROUP BY c.country ORDER BY avg_order_spend DESC;"),
    ("How many Accessories products are there?",
     "SELECT COUNT(*) AS n FROM products WHERE category = 'Accessories';"),
    ("List the email addresses of customers from Paris.",
     "SELECT name, email FROM customers WHERE city = 'Paris' ORDER BY name;"),
    ("Which orders include the product 'Laptop Pro 15'?",
     "SELECT oi.order_id, o.order_date, o.status "
     "FROM order_items oi "
     "JOIN products p ON oi.product_id = p.product_id "
     "JOIN orders o ON oi.order_id = o.order_id "
     "WHERE p.name = 'Laptop Pro 15' ORDER BY o.order_date;"),
    ("Show total revenue by customer country.",
     "SELECT c.country, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY c.country ORDER BY revenue DESC;"),
    ("Find the highest-revenue single order.",
     "SELECT order_id, ROUND(SUM(quantity * unit_price), 2) AS revenue "
     "FROM order_items GROUP BY order_id ORDER BY revenue DESC LIMIT 1;"),
    ("How many Stationery products cost under $10?",
     "SELECT COUNT(*) AS n FROM products "
     "WHERE category = 'Stationery' AND price < 10;"),
    ("List products that have been ordered and still have more than 50 in stock.",
     "SELECT DISTINCT p.name, p.stock_qty "
     "FROM products p "
     "JOIN order_items oi ON p.product_id = oi.product_id "
     "WHERE p.stock_qty > 50 ORDER BY p.stock_qty DESC;"),

    # ---- even more variety ----
    ("Show each customer's first order date.",
     "SELECT customer_id, MIN(order_date) AS first_order "
     "FROM orders GROUP BY customer_id ORDER BY first_order LIMIT 10;"),
    ("Count orders by status for the year 2023.",
     "SELECT status, COUNT(*) AS n FROM orders "
     "WHERE order_date LIKE '2023%' GROUP BY status ORDER BY status;"),
    ("Which product categories have an average price above $100?",
     "SELECT category, ROUND(AVG(price), 2) AS avg_price "
     "FROM products GROUP BY category HAVING avg_price > 100 ORDER BY avg_price DESC;"),
    ("Show the 10 customers who spent the most overall.",
     "SELECT c.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS total_spent "
     "FROM customers c "
     "JOIN orders o ON c.customer_id = o.customer_id "
     "JOIN order_items oi ON o.order_id = oi.order_id "
     "GROUP BY c.customer_id ORDER BY total_spent DESC LIMIT 10;"),
    ("What is the total quantity sold for Furniture products?",
     "SELECT SUM(oi.quantity) AS qty_sold "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "WHERE p.category = 'Furniture';"),
    ("Find the average stock quantity across all products.",
     "SELECT ROUND(AVG(stock_qty), 1) AS avg_stock FROM products;"),
    ("List the distinct statuses used in the orders table.",
     "SELECT DISTINCT status FROM orders ORDER BY status;"),
    ("Show total sales (revenue) per product, top 10.",
     "SELECT p.name, ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.product_id ORDER BY revenue DESC LIMIT 10;"),
    ("How many orders did each customer place? Show all customers.",
     "SELECT c.name, COUNT(o.order_id) AS n_orders "
     "FROM customers c LEFT JOIN orders o ON c.customer_id = o.customer_id "
     "GROUP BY c.customer_id ORDER BY n_orders DESC LIMIT 20;"),
    ("Which products have never been ordered?",
     "SELECT name FROM products "
     "WHERE product_id NOT IN (SELECT DISTINCT product_id FROM order_items) "
     "ORDER BY name;"),

    # ---- last batch ----
    ("What is the most common order status?",
     "SELECT status, COUNT(*) AS n FROM orders GROUP BY status ORDER BY n DESC LIMIT 1;"),
    ("Show the average price of Electronics products.",
     "SELECT ROUND(AVG(price), 2) AS avg_price FROM products "
     "WHERE category = 'Electronics';"),
    ("How many customers are from each country?",
     "SELECT country, COUNT(*) AS n FROM customers GROUP BY country ORDER BY n DESC;"),
    ("Find orders placed by customers from Italy.",
     "SELECT o.order_id, c.name, o.order_date, o.status "
     "FROM orders o JOIN customers c ON o.customer_id = c.customer_id "
     "WHERE c.country = 'Italy' ORDER BY o.order_date DESC LIMIT 10;"),
    ("Show the total quantity ordered per product category.",
     "SELECT p.category, SUM(oi.quantity) AS qty "
     "FROM order_items oi JOIN products p ON oi.product_id = p.product_id "
     "GROUP BY p.category ORDER BY qty DESC;"),
    ("Which customers placed orders in both 2022 and 2023?",
     "SELECT customer_id FROM orders WHERE order_date LIKE '2022%' "
     "INTERSECT "
     "SELECT customer_id FROM orders WHERE order_date LIKE '2023%' "
     "ORDER BY customer_id;"),
    ("What is the revenue from shipped orders?",
     "SELECT ROUND(SUM(oi.quantity * oi.unit_price), 2) AS revenue "
     "FROM order_items oi JOIN orders o ON oi.order_id = o.order_id "
     "WHERE o.status = 'shipped';"),
    ("How many orders were placed in each quarter of 2023?",
     "SELECT "
     "CASE WHEN order_date BETWEEN '2023-01-01' AND '2023-03-31' THEN 'Q1' "
     "     WHEN order_date BETWEEN '2023-04-01' AND '2023-06-30' THEN 'Q2' "
     "     WHEN order_date BETWEEN '2023-07-01' AND '2023-09-30' THEN 'Q3' "
     "     ELSE 'Q4' END AS quarter, "
     "COUNT(*) AS n "
     "FROM orders WHERE order_date LIKE '2023%' "
     "GROUP BY quarter ORDER BY quarter;"),
    ("List all products in the Appliances category with their prices.",
     "SELECT name, price FROM products WHERE category = 'Appliances' ORDER BY price;"),
    ("Show the 3 most recently joined customers.",
     "SELECT name, joined_date FROM customers ORDER BY joined_date DESC LIMIT 3;"),
]


def validate_and_write(pairs, conn):
    """Execute every SQL; raise immediately on failure. Return validated list."""
    validated = []
    for q, sql in pairs:
        try:
            rows = conn.execute(sql).fetchall()
            _ = rows  # force execution
        except Exception as exc:
            raise RuntimeError(f"SQL failed:\n  Q: {q}\n  SQL: {sql}\n  Error: {exc}") from exc
        validated.append({"question": q, "schema": SCHEMA_TEXT, "sql": sql})
    return validated


def write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    print(f"Wrote {len(records)} pairs -> {path}")


def main():
    build_db()
    conn = sqlite3.connect(DB_PATH)

    all_pairs = list(PAIRS)
    rng2 = random.Random(SEED)
    rng2.shuffle(all_pairs)

    n_total = len(all_pairs)
    n_heldout = 40
    n_train = n_total - n_heldout
    train_pairs = all_pairs[:n_train]
    heldout_pairs = all_pairs[n_train:]

    print(f"Total pairs: {n_total}  |  train: {n_train}  |  heldout: {n_heldout}")

    print("Validating train SQL…")
    train_records = validate_and_write(train_pairs, conn)
    print("Validating heldout SQL…")
    heldout_records = validate_and_write(heldout_pairs, conn)

    conn.close()

    write_jsonl(TRAIN_PATH, train_records)
    write_jsonl(HELDOUT_PATH, heldout_records)
    print("All SQL validated against data/sample.sqlite. Done.")


if __name__ == "__main__":
    main()
