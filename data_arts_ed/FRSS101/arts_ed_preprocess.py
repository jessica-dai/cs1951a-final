import numpy as np
import sys
  
def get_score(subarr):
    for elt in subarr: 
        if (elt not in [" ", "-", "8"]):
            return int(elt)
    return 0

def get_key(c, r):
    if c == 1 and r == 1:
        return 1
    if c == 1 and r == 2:
        return 2
    if c == 1 and r == 3:
        return 3
    if c == 1 and r == 4:
        return 4
    if c == 2 and r == 1:
        return 5
    if c == 2 and r == 2:
        return 6
    if c == 2 and r == 3:
        return 7
    if c == 2 and r == 4:
        return 8
    if c == 3 and r == 1:
        return 9
    if c == 3 and r == 2:
        return 10
    if c == 3 and r == 3:
        return 11
    if c == 3 and r == 4:
        return 12
    if c == 4 and r == 1:
        return 13
    if c == 4 and r == 2:
        return 14
    if c == 4 and r == 3:
        return 15
    if c == 4 and r == 4:
        return 16

def get_rows():
    read_data = []
    with open('frss101.dat', 'r') as datafile:
        for line in datafile:
            read_data.append(line)
    processed_data = []

    for line in read_data:

        idn = int(line[0:4])
        enroll_size = int(line[5])
        c = int(line[6])
        r = int (line[7])
        vis_art_taught = int (line[12])
        music_taught = int (line[13])
        dance_taught = int (line[14])
        drama_taught = int (line[15])
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
        key = get_key(c,r)
        dist_data = [enroll_size, 
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
                    key
                    ]
        processed_data.append(dist_data)
    return processed_data

        
        