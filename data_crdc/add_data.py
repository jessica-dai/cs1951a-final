import sqlite3
from crdc_preprocess import get_crdc_rows

database = "../all_data.db"

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

crdc_sql_statement = """ INSERT INTO crdc(
				dist_state_abbrev,
				dist_state_name,
				dist_id, 
				dist_name,
				school_id,
				school_name,
				dist_and_school_id,
				ov_enr_hisp_m,
				ov_enr_hisp_f,
				ov_enr_ai_an_m,
				ov_enr_ai_an_f,
				ov_enr_as_m,
				ov_enr_as_f,
				ov_enr_nh_m,
				ov_enr_nh_f,
				ov_enr_b_m,
				ov_enr_b_f,
				ov_enr_w_m,
				ov_enr_w_f,
				ov_enr_twomr_m,
				ov_enr_twom_f,
				g_and_t,
				num_adv_ma,
				ap_ind,
				num_ap_c,
				tot_ap_pass_m,
				tot_ap_pass_f,
				ib_ind,
				har_bul_sex,
				har_bul_race,
				har_bul_dis,
				har_bul_sex_o,
				har_bul_rel,
				region)
			VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
	"""

for row in get_crdc_rows():
    add_data(cur, row, crdc_sql_statement)