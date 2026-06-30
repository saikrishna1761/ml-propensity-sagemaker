import boto3
import json
import psycopg2
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker()
random.seed(42)
np.random.seed(42)

NUM_CUSTOMERS = 10_000
NUM_ORDERS = 40_000
SNAPSHOT_DATE = datetime(2026, 6, 1)

PRODUCT_CATEGORIES = ["electronics", "clothing", "groceries", "books", "home", "beauty", "sports"]
PAYMENT_METHODS = ["card", "upi", "cod", "netbanking", "wallet"]
ORDER_STATUSES = ["completed", "cancelled", "returned"]
ORDER_STATUS_WEIGHTS = [0.80, 0.12, 0.08]
ACCOUNT_STATUSES = ["active", "inactive"]


def get_db_credentials():
    client = boto3.client("secretsmanager", region_name="ap-south-1")
    secret = client.get_secret_value(SecretId="ml/postgres")
    return json.loads(secret["SecretString"])


def get_connection(creds):
    return psycopg2.connect(
        host=creds["host"],
        port=creds["port"],
        dbname=creds["dbname"],
        user=creds["username"],
        password=creds["password"],
    )


def create_tables(cursor):
    cursor.execute("""
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id     SERIAL PRIMARY KEY,
            email           VARCHAR(255) UNIQUE NOT NULL,
            age             INTEGER,
            city            VARCHAR(100),
            country         VARCHAR(100),
            account_status  VARCHAR(20),
            email_opens_30d INTEGER,
            signup_date     DATE
        );

        CREATE TABLE orders (
            order_id         SERIAL PRIMARY KEY,
            customer_id      INTEGER REFERENCES customers(customer_id),
            order_date       TIMESTAMP,
            order_value      NUMERIC(10, 2),
            product_category VARCHAR(50),
            payment_method   VARCHAR(20),
            order_status     VARCHAR(20)
        );
    """)


def generate_customers():
    customers = []
    for _ in range(NUM_CUSTOMERS):
        customers.append((
            fake.unique.email(),
            int(np.random.randint(18, 70)),
            fake.city(),
            fake.country(),
            random.choices(ACCOUNT_STATUSES, weights=[0.85, 0.15])[0],
            int(np.random.poisson(lam=5)),                          # most open ~5 emails
            fake.date_between(start_date="-5y", end_date="-30d"),
        ))
    return customers


def generate_orders(customer_ids):
    orders = []
    # distribute 40k orders across customers with realistic skew
    order_counts = np.random.negative_binomial(2, 0.3, size=NUM_CUSTOMERS)
    order_counts = np.clip(order_counts, 0, 50)

    # scale to exactly NUM_ORDERS
    scale = NUM_ORDERS / order_counts.sum()
    order_counts = np.round(order_counts * scale).astype(int)
    diff = NUM_ORDERS - order_counts.sum()
    order_counts[np.argmax(order_counts)] += diff

    for cid, count in zip(customer_ids, order_counts):
        for _ in range(count):
            order_date = SNAPSHOT_DATE - timedelta(days=int(np.random.exponential(scale=60)))
            order_date = max(order_date, SNAPSHOT_DATE - timedelta(days=365))
            orders.append((
                cid,
                order_date,
                round(float(np.random.lognormal(mean=4.0, sigma=1.0)), 2),  # realistic order values
                random.choice(PRODUCT_CATEGORIES),
                random.choice(PAYMENT_METHODS),
                random.choices(ORDER_STATUSES, weights=ORDER_STATUS_WEIGHTS)[0],
            ))
    return orders


def insert_customers(cursor, customers):
    cursor.executemany("""
        INSERT INTO customers (email, age, city, country, account_status, email_opens_30d, signup_date)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, customers)


def insert_orders(cursor, orders):
    cursor.executemany("""
        INSERT INTO orders (customer_id, order_date, order_value, product_category, payment_method, order_status)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, orders)


def main():
    print("Fetching credentials from Secrets Manager...")
    creds = get_db_credentials()

    print("Connecting to Postgres...")
    conn = get_connection(creds)
    conn.autocommit = False
    cursor = conn.cursor()

    print("Creating tables...")
    create_tables(cursor)
    conn.commit()

    print(f"Generating {NUM_CUSTOMERS} customers...")
    customers = generate_customers()
    insert_customers(cursor, customers)
    conn.commit()

    print("Fetching customer IDs...")
    cursor.execute("SELECT customer_id FROM customers ORDER BY customer_id")
    customer_ids = [row[0] for row in cursor.fetchall()]

    print(f"Generating {NUM_ORDERS} orders...")
    orders = generate_orders(customer_ids)
    insert_orders(cursor, orders)
    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM customers")
    print(f"Customers inserted: {cursor.fetchone()[0]}")

    cursor.execute("SELECT COUNT(*) FROM orders")
    print(f"Orders inserted: {cursor.fetchone()[0]}")

    cursor.close()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
