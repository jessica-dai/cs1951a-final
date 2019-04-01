read_data = []
with open('FRSS108PUF.dat', 'r') as datafile:
    for line in datafile:
        read_data.append(line)

# each line has 1442 chars, 1527 lines total

# quality measures? sum of 10-77
# barrier measures? 80-93 (providing), 96-111 (student participation)
# resource limitations for adding + phasing out? 118-123 (adding), 138-143 (removing)
# all of the above aren't zero-indexed (based on key provided w dataset)

processed_data = []

def get_score(subarr):
    score = 0
    for elt in subarr: 
        if (elt not in [" ", "-", "8"]):
            score += int(elt)
    return score

rsc_add_nulls = 0
rsc_rmv_nulls = 0
add_rmv_nulls = 0

for line in read_data:

    idn = int(line[0:4])

    # only add to processed data if line[9] == 1 (flag for y/n CTE in HS)
    if (line[8] == "1"):

        # joinable attributes
        dist_size = line[5] # 1 = less than 2k, 2 = 2k < 5k, 3 = 5k+
        urb = line[6] # 1 = city, 2 = suburban, 3 = town, 4 = rural
        region = line[7] # 1 = northeast, 2 = southeast, 3 = central, 4 = west

        # response vars
        prog_qual = get_score(line[10:77])
        barriers_providing = get_score(line[80:93])
        barriers_participation = get_score(line[96:111])
        resources_adding = get_score(line[118:123])
        resources_removing = get_score(line[138:143])

        # add to processed_data
        dist_data = [idn, dist_size, urb, region, prog_qual, barriers_providing, barriers_participation, resources_adding, resources_removing]
        processed_data.append(dist_data)

        # count nulls (zero nulls for prog_qual, barriers_providing, barriers_participation)
        if (resources_adding == 0):
            rsc_add_nulls += 1

        if (resources_removing == 0):
            rsc_rmv_nulls += 1
        
        if (resources_adding == 0 and resources_removing == 0):
            add_rmv_nulls += 1

        # TODO add to sql database?

print(len(processed_data)) # = 1510 -- we removed 17 records
print(len(processed_data[0])) # = 9 -- 0: ID, 1-3: joinable attributes, 4-7: response variables

print(rsc_add_nulls) # = 152 
print(rsc_rmv_nulls) # = 152 
print(add_rmv_nulls) # = 152 --> 152 districts didn't respond 