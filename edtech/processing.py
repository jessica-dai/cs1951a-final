import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sqlite3 import Error

database = "/Users/rebeccazuo/Desktop/cs1951a-final/all_data.db"

columns = ('id', 'dist_size', 'urb', 'region', 'totalComputers', 'computersForInstruction', 'integration', 'training')

def create_connection(db_file):
	try:
		conn = sqlite3.connect(db_file)
		return conn
	except:
		print("ERROR: Could not create connection.")

conn = create_connection(database)
cur = conn.cursor()


def create_table(conn, sql_text):
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
        c.close()
    except Error as e:
        print(e)



create_table_sql = """ CREATE TABLE IF NOT EXISTS edTech (
					combined integer,
                    dist_size integer,
                    urb integer,
                    region integer,
                    totalComputers integer NOT NULL ,
                    computersForInstruction integer ,
                    integration integer,
                    training integer
                ); """


# add_column = "ALTER TABLE edTech ADD id integer;"

sql = "DROP TABLE edTech"
cur.execute(sql)

create_table(conn,create_table_sql)


insert_sql  = """ INSERT INTO edTech(
	combined,
    dist_size ,
    urb ,
    region ,
    totalComputers  ,
    computersForInstruction  ,
    integration ,
    training )

    VALUES(?,?,?,?,?,?,?,?)
    """



connection = create_connection(database)

if connection is not None:
    create_table(connection, database)
else:
    print("Not connecting")


read_data = []
processed_data = []
with open('frss92.dat', 'r') as datafile:
    for line in datafile:
        read_data.append(line)


def get_score(subarr):
    score = 0
    for elt in subarr:
        if (elt not in [" ", "-", "8"]):
            score += int(elt)
    return score

def add_data(cur, arr, sql_statement):
    cur.execute(sql_statement, arr)
    conn.commit()
    return cur.lastrowid


for line in read_data:

    idn = int(line[1:5])

    if(line[6] == "2"):

        # joinable attributes
        dist_size = line[7] # 1 = less than 2500k, 2 = 2500k < 9999k, 3 = 10000k+
        urb = int(line[8]) # 1 = city, 2 = suburban, 3 = town, 4 = rural
        region = int(line[9]) # 1 = northeast, 2 = southeast, 3 = central, 4 = west

        # response vars
        totalComputers = get_score(line[16:27]) #
        computersForInstruction = get_score(line[24:27]) #
        integration = get_score(line[135]) # on a range of 1-4 if district staff help technology integration
        training = get_score(line[154]) # on a range of 1-4 are teachers sufficiently trained to integrate technology
        if (urb == 1 and region == 1):
	        combined = 1
        elif(urb == 1 and region == 2):
	        combined = 2
        elif(urb== 1 and region == 3):
	        combined = 3
        elif(urb== 1 and region == 4):
	        combined = 4
        elif(urb == 2 and region == 1):
	        combined = 5
        elif(urb == 2 and region == 2):
	        combined = 6
        elif(urb == 2 and region == 3):
	        combined = 7
        elif(urb == 2 and region == 4):
	        combined = 8
        elif(urb == 3 and region == 1):
	        combined = 9
        elif(urb == 3 and region == 2):
	        combined = 10
        elif(urb == 3 and region == 3):
	        combined = 11
        elif(urb == 3 and region == 4):
	        combined = 12
        elif(urb == 4 and region == 1):
	        combined = 13
        elif(urb == 4 and region == 2):
	        combined = 14
        elif(urb == 4 and region == 3):
	        combined = 15
        elif(urb == 4 and region == 4):
	        combined = 16
        # add to processed_data
        dist_data = [combined, dist_size, urb, region, totalComputers, computersForInstruction, integration, training]
        processed_data.append(dist_data)
        x = add_data(cur, dist_data, insert_sql)




conn.close()
# no nulls in data

#print(len(processed_data)) # = 916
#print(len(processed_data[0])) # = 7
