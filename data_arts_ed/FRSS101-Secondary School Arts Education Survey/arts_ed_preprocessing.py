import numpy as np
import sqlite3

database = "../all_data.db"

def create_connection(db_file):

    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None

def get_score(subarr):
    for elt in subarr: 
        if (elt not in [" ", "-", "8"]):
            return int(elt)
    return 0

def add_data(cur, arr):

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
                drama_teachers)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    cur.execute(sql, arr)
    conn.commit()
    return cur.lastrowid


conn = create_connection(database)
cur = conn.cursor()

read_data = []
with open('frss101.dat', 'r') as datafile:
    for line in datafile:
        read_data.append(line)


processed_data = []

for line in read_data:
    
    idn = int(line[0:4])
    enroll_size = line[5]
    comm_type = line[6]
    region = line[7]
    vis_art_taught = line[12]
    music_taught = line[13]
    dance_taught = line[14]
    drama_taught = line[15]
    vis_a_courses = get_score(line[16:17])
    music_courses = get_score(line[18:19])
    dance_courses = get_score(line[20:21])
    drama_courses = get_score(line[22:23])
    vis_a_students = get_score(line[24:27])
    music_students = get_score(line[28:31])
    dance_students= get_score(line[32:35]) 
    drama_students= get_score(line[36:39]) 
    vis_a_teachers = get_score(line[40:41])
    music_teachers = get_score(line[42:43])
    dance_teachers = get_score(line[44:45])
    drama_teachers = get_score(line[46:47])
    dist_data = [idn, enroll_size, 
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
                drama_teachers]
    processed_data.append(dist_data)
    x = add_data(cur, dist_data[1:])
    print (x)
    
conn.close()
