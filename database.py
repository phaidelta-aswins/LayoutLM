import sqlite3

# Database initialization
def init_db():
    conn = sqlite3.connect("ocr_data1.db")
    cursor = conn.cursor()

    # Create a table to store OCR results
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS extracted_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT,
            word TEXT,
            tag TEXT,
            bbox TEXT
        )
    ''')
    conn.commit()
    conn.close()

# Function to insert extracted data into the database
def insert_data(image_name, words, tags, bboxes):
    conn = sqlite3.connect("ocr_data1.db")
    cursor = conn.cursor()

    for word, tag, bbox in zip(words, tags, bboxes):
        cursor.execute("INSERT INTO extracted_data (image_name, word, tag, bbox) VALUES (?, ?, ?, ?)", 
                       (image_name, word, tag, str(bbox)))

    conn.commit()
    conn.close()

# Initialize the database
init_db()
