import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import csv
import sqlite3
import sys


ne = ["Connecticut", "Delaware", "District of Columbia", "Maine", "Maryland", "Massachusetts", "New Hampshire", "New Jersey", "New York", "Pennsylvania", "Rhode Island", "Vermont"]
se = ["Alabama", "Arkansas", "Florida", "Georgia", "Kentucky", "Louisiana", "Mississippi", "North Carolina", "South Carolina", "Tennessee", "Virginia", "West Virginia"]
ce = ["Illinois", "Indiana", "Iowa", "Kansas", "Michigan", "Minnesota", "Missouri", "Nebraska", "North Dakota", "Ohio", "South Dakota", "Wisconsin"]
we = ["Alaska", "Arizona", "California", "Colorado", "Hawaii", "Idaho", "Montana", "Nevada", "New Mexico", "Oklahoma", "Oregon", "Texas", "Utah", "Washington", "Wyoming"]

ne = list(map(lambda x: x.upper(), ne))
se = list(map(lambda x: x.upper(), se))
ce = list(map(lambda x: x.upper(), ce))
we = list(map(lambda x: x.upper(), we))

def get_crdc_rows():
	with open('/users/amypu/documents/Data_Files_and_Layouts/CRDC_2015-16_School_Data.csv') as data:
		read_data = csv.reader(data)
		processed_data = []
		for line in read_data:
			# only add to processed data if line[18] == Yes and line[19] == Yes and line[20] == Yes and line[21] == Yes
			if (line[18] == "Yes") and (line[19] == "Yes") and (line[20] == "Yes") and (line[21] == "Yes"):
				# district_state_abbrev = line[0] #district state abbreviation
				district_state_name = line[1] #district state abbreviation
				# district_id = line[2] #7 digit district id
				# district_name = line[3] #district name
				# school_id = line[4] #5 digit school id
				# school_name = line[5] #school name
				# district_and_school_id = line[6] #7 digit district and 5 digit school id
				overall_enr_hisp_m = line[55] #overall student enrollment: hispanic male
				overall_enr_hisp_f = line[56] #overall student enrollment: hispanic female
				overall_enr_ai_an_m = line[57] #overall student enrollment: american indian/alaska native male
				overall_enr_ai_an_f = line[58] #overall student enrollment: american indian/alaska native female
				overall_enr_as_m = line[59] #overall student enrollment: asian male
				overall_enr_as_f = line[60] #overall student enrollment: asian female
				overall_enr_nh_m = line[61] #overall student enrollment: native hawaiian/pacific islander male
				overall_enr_nh_f = line[62] #overall student enrollment: native hawaiian/pacific islander female
				overall_enr_b_m = line[63] #overall student enrollment: black male
				overall_enr_b_f = line[64] #overall student enrollment: black female
				overall_enr_w_m = line[65] #overall student enrollment: white male
				overall_enr_w_f = line[66] #overall student enrollment: white female
				overall_enr_twomr_m = line[67] #overall student enrollment: two or more races male
				overall_enr_twomr_f = line[68] #overall student enrollment: two or more races female
				overall_enr_m_tot = line[69] #overall student enrollment: calculated male total
				overall_enr_f_tot = line[70] #overall student enrollment: calculated female total

				# g_and_t = line[147] #gifted and talented indicator

				# num_adv_ma = line[387] #number of advanced mathematics classes

				# ap_ind = line[491] #AP indicator: does this school have any students in AP programs?

				# num_ap_c = line[492] #number of different AP courses offered

				# tot_ap_pass_m = line[631] #number of students who passed some AP exams: male
				# tot_ap_pass_f = line[632] #number of students who passed some AP exams: female

				# ib_ind = line[657] #IB program indicator

				# har_bul_sex = line[1327] #allegations of harrassment or bullying on the basis of sex
				# har_bul_race = line[1328] #allegations of harrassment or bullying on the basis of race, color, or national origin
				# har_bul_dis = line[1329] #allegations of harrassment or bullying on the basis of disability
				# har_bul_sex_o = line[1330] #allegations of harrassment or bullying on the basis of sexual orientation
				# har_bul_rel = line[1331] #allegations of harrassment or bullying on the basis of religion

				if district_state_name in ne:
					region = 0
				elif district_state_name in se:
					region = 1
				elif district_state_name in ce:
					region = 2
				else:
					region = 3

				# row = [ \
				# 	# district_state_abbrev, \
				# 	# district_state_name, \
				# 	# district_id, \
				# 	# district_name, \
				# 	# school_id, \
				# 	# school_name, \
				# 	# district_and_school_id, \
				# 	overall_enr_hisp_m, \
				# 	overall_enr_hisp_f, \
				# 	overall_enr_ai_an_m, \
				# 	overall_enr_ai_an_f, \
				# 	overall_enr_as_m, \
				# 	overall_enr_as_f, \
				# 	overall_enr_nh_m, \
				# 	overall_enr_nh_f, \
				# 	overall_enr_b_m, \
				# 	overall_enr_b_f, \
				# 	overall_enr_w_m, \
				# 	overall_enr_w_f, \
				# 	overall_enr_twomr_m, \
				# 	overall_enr_twomr_f, \
				# 	# g_and_t, \
				# 	# num_adv_ma, \
				# 	# ap_ind, \
				# 	# num_ap_c, \
				# 	# tot_ap_pass_m, \
				# 	# tot_ap_pass_f, \
				# 	# ib_ind, \
				# 	# har_bul_sex, \
				# 	# har_bul_race, \
				# 	# har_bul_dis, \
				# 	# har_bul_sex_o, \
				# 	# har_bul_rel, \
				# 	region
				# ]
				row = [ \
					int(overall_enr_hisp_m), \
					int(overall_enr_hisp_f), \
					int(overall_enr_ai_an_m), \
					int(overall_enr_ai_an_f), \
					int(overall_enr_as_m), \
					int(overall_enr_as_f), \
					int(overall_enr_nh_m), \
					int(overall_enr_nh_f), \
					int(overall_enr_b_m), \
					int(overall_enr_b_f), \
					int(overall_enr_w_m), \
					int(overall_enr_w_f), \
					int(overall_enr_twomr_m), \
					int(overall_enr_twomr_f), \
					int(overall_enr_m_tot), \
					int(overall_enr_f_tot), \
					int(region)
				]

				processed_data.append(row)
				
		return processed_data

def get_as_numpy():
	return np.array(get_crdc_rows())