import sqlite3
import csv

# Database connection
db_file = "species.db"  # Ensure this matches your database file name
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# Function to populate a table from a CSV file
def populate_table(csv_file, table_name, columns):
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            print(f"Row: {row}, Length: {len(row)}")
            placeholders = ', '.join(['?'] * len(row))
            query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
            print(f"Query: {query}")
            cursor.execute(query, row)
    print(f"Data from {csv_file} has been inserted into {table_name}.")

# Populate animals table
#populate_table("animals.csv", "animals", "id, name, scientific_name, description, habitat, diet")

# Populate plants table
#populate_table("plants.csv", "plants", "id, name, scientific_name, description, habitat, flowering_season")

# Commit changes and close connection
conn.commit()
conn.close()

print("Database population completed successfully!")

