import sqlite3

database = "../all_data.db"

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

create_crdc_table_sql = """ CREATE TABLE IF NOT EXISTS crdc (
					id integer PRIMARY KEY,
					ov_enr_hisp_m integer,
					ov_enr_hisp_f integer,
					ov_enr_ai_an_m integer,
					ov_enr_ai_an_f integer,
					ov_enr_as_m integer,
					ov_enr_as_f integer,
					ov_enr_nh_m integer,
					ov_enr_nh_f integer,
					ov_enr_b_m integer,
					ov_enr_b_f integer,
					ov_enr_w_m integer,
					ov_enr_w_f integer,
					ov_enr_twomr_m integer,
					ov_enr_twom_f integer,
					overall_enr_m_tot integer,
					overall_enr_f_tot integer,
					region integer
				); """

# create_crdc_table_sql = """ CREATE TABLE IF NOT EXISTS crdc (
# 					id integer PRIMARY KEY,
# 					# dist_state_abbrev TEXT, 
# 					# dist_state_name TEXT, 
# 					# dist_id integer,
# 					# dist_name TEXT, 
# 					# school_id integer,
# 					# school_name TEXT,
# 					# dist_and_school_id integer,
# 					ov_enr_hisp_m integer,
# 					ov_enr_hisp_f integer,
# 					ov_enr_ai_an_m integer,
# 					ov_enr_ai_an_f integer,
# 					ov_enr_as_m integer,
# 					ov_enr_as_f integer,
# 					ov_enr_nh_m integer,
# 					ov_enr_nh_f integer,
# 					ov_enr_b_m integer,
# 					ov_enr_b_f integer,
# 					ov_enr_w_m integer,
# 					ov_enr_w_f integer,
# 					ov_enr_twomr_m integer,
# 					ov_enr_twom_f integer,
# 					# g_and_t TEXT,
# 					# num_adv_ma integer,
# 					# ap_ind TEXT, 
# 					# num_ap_c integer,
# 					# tot_ap_pass_m integer,
# 					# tot_ap_pass_f integer,
# 					# ib_ind TEXT,
# 					# har_bul_sex integer,
# 					# har_bul_race integer,
# 					# har_bul_dis integer,
# 					# har_bul_sex_o integer,
# 					# har_bul_rel integer,
# 					region integer
# 				); """

if conn is not None:
	create_table(conn, create_crdc_table_sql)
else:
	print("ERROR: Connection is null.")