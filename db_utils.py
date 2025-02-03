import sqlite3

def get_db_connection():
    db_file = "species.db"
    connection = sqlite3.connect(db_file)
    connection.row_factory = sqlite3.Row  # Fetch rows as dictionaries
    return connection

def get_all_animals():
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM animals"
    cursor.execute(query)
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_all_plants():
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM plants"
    cursor.execute(query)
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_animal_by_id(class_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM animals WHERE id = ?"
    print(f"Executing query: {query} with class_id: {class_id} (type: {type(class_id)})")
    cursor.execute(query, (class_id,))
    animal = cursor.fetchone()
    print(f"Raw query result: {animal}")
    connection.close()
    return dict(animal) if animal else None

def get_plant_by_id(class_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM plants WHERE id = ?"
    print(f"Executing query: {query} with class_id: {class_id}")
    cursor.execute(query, (class_id,))
    plant = cursor.fetchone()
    print(f"Raw query result: {plant}")
    connection.close()
    return dict(plant) if plant else {"error": f"No plant found with ID {class_id}"}

def get_species_by_id(class_id):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Check in the animals table
    query_animal = "SELECT * FROM animals WHERE id = ?"
    cursor.execute(query_animal, (class_id,))
    animal = cursor.fetchone()

    if animal:
        connection.close()
        return {"type": "animal", "data": dict(animal)}

    # Check in the plants table
    query_plant = "SELECT * FROM plants WHERE id = ?"
    cursor.execute(query_plant, (class_id,))
    plant = cursor.fetchone()

    connection.close()
    if plant:
        return {"type": "plant", "data": dict(plant)}
    
    return {"error": f"No species found with ID {class_id}"}

def search_species_by_name(name):
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Exact match query for animals
    query_exact_animal = "SELECT * FROM animals WHERE name = ?"
    cursor.execute(query_exact_animal, (name,))
    exact_animals = cursor.fetchall()
    
    # Exact match query for plants
    query_exact_plant = "SELECT * FROM plants WHERE name = ?"
    cursor.execute(query_exact_plant, (name,))
    exact_plants = cursor.fetchall()
    
    # If no exact match, look for partial matches
    if not exact_animals and not exact_plants:
        query_partial_animal = "SELECT * FROM animals WHERE name LIKE ?"
        cursor.execute(query_partial_animal, (f"%{name}%",))
        partial_animals = cursor.fetchall()
        
        query_partial_plant = "SELECT * FROM plants WHERE name LIKE ?"
        cursor.execute(query_partial_plant, (f"%{name}%",))
        partial_plants = cursor.fetchall()
        
        connection.close()
        return {
            "exact_matches": [],
            "partial_matches": {
                "animals": [dict(animal) for animal in partial_animals],
                "plants": [dict(plant) for plant in partial_plants]
            }
        }

    # Return exact matches
    connection.close()
    return {
        "exact_matches": {
            "animals": [dict(animal) for animal in exact_animals],
            "plants": [dict(plant) for plant in exact_plants]
        },
        "partial_matches": {}}
    

# animal_info = search_species_by_name('cate')
# print(animal_info)