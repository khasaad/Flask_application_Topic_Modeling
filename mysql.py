import sqlite3

connection = sqlite3.connect('database.db')

if connection:
    print("Connected Successfully")
else:
    print("Connection Not Established")

cursor = connection.cursor()

# Create topic_modeling table

# command_table = '''CREATE TABLE IF NOT EXISTS
#     topic_modeling(article message_text , USA_politic REAL,
#     General_security REAL, Entertainment REAL, Health_care REAL, Elections REAL, Women_rights REAL, Well_being REAL,
#      Education REAL, Social_security REAL, World_politics REAL);'''
#
# cursor.execute(command_table)

# cursor.execute('DROP TABLE topic_modeling;',)

# cursor.execute("INSERT INTO topic_modeling VALUES ('article 1', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9)")

print(cursor.execute("SELECT * FROM topic_modeling"))

rows = cursor.fetchall()

for row in rows:
    print(row)
# Save
# connection.commit()

# Closing the connection
# connection.close()