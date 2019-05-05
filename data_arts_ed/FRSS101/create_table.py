import sqlite3

database = "../../all_data.db"

def create_connection(db_file):
	try:
		conn = sqlite3.connect(db_file)
		return conn
	except:
		print("ERROR: Could not create connection.")

conn = create_connection(database)
cur = conn.cursor()

def create_table(conn, sql_statement):
	try:
		c = conn.cursor()
		c.execute(sql_statement)
		c.close()
	except:
		print("ERROR: Could not create CRDC table.")

create_artsed_table_sql = """ CREATE TABLE IF NOT EXISTS artsed (
					id integer PRIMARY KEY,
                    enroll_size integer, 
                comm_type integer, 
                region integer, 
                vis_art_taught integer,
                music_taught integer,
                dance_taught integer,
                drama_taught integer,
                vis_a_courses integer,
                music_courses integer,
                dance_courses integer,
                drama_courses integer,
                vis_a_students integer,
                music_students integer,
                dance_students integer,
                drama_students integer,
                vis_a_teachers integer,
                music_teachers integer,
                dance_teachers integer,
                drama_teachers integer,
                key integer
                ); """

if conn is not None:
	create_table(conn, create_artsed_table_sql)
else:
	print("ERROR: Connection is null.")
