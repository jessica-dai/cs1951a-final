import sqlite3

basequery = "select count(urb) from cte where "
above_avg_qual = "prog_qual>73"
urb_wheres = [
    "urb=1",
    "urb=2",
    "urb=3",
    "urb=4",
]


def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def qual_analysis(conn):
    cur = conn.cursor()

    barrier_whe

def barriers_analysis(conn):
    cur = conn.cursor()

    basequery = "select count(urb) from cte where "

def main():
    database="../all_data.db"
    conn = create_connection(database)

    quality = qual_analysis

    with conn:
        quality = qual_analysis(conn)
    
