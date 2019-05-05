import numpy as np
import sqlite3
from arts_ed_preprocess import get_rows

database = "../../all_data.db"

def create_connection(db_file):
	try:
		conn = sqlite3.connect(db_file)
		return conn
	except:
		print("ERROR: Could not create connection.")

conn = create_connection(database)
cur = conn.cursor()

def add_data(cur, arr, sql_statement):
	cur.execute(sql_statement, arr)
	conn.commit()

sql = """ INSERT INTO artsed(
            enroll_size, 
            comm_type, 
            region, 
            vis_art_taught,
            music_taught,
            dance_taught,
            drama_taught,
            vis_a_courses,
            music_courses,
            dance_courses,
            drama_courses,
            vis_a_students,
            music_students,
            dance_students,
            drama_students,
            vis_a_teachers,
            music_teachers,
            dance_teachers,
            drama_teachers,
            key)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""

for row in get_rows():
    add_data(cur, row, sql)
    