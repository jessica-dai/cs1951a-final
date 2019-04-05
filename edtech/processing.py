read_data = []
with open('frss93.dat', 'r') as datafile:
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

nulls1 = 0
nulls2 = 0
nulls3 = 0
nulls4 = 0

add_rmv_nulls = 0

for line in read_data:

    idn = int(line[0:4])

    if(line[6] == "2"):

        # joinable attributes
        dist_size = line[7] # 1 = less than 2500k, 2 = 2500k < 9999k, 3 = 10000k+
        urb = line[8] # 1 = city, 2 = suburban, 3 = town, 4 = rural
        region = line[9] # 1 = northeast, 2 = southeast, 3 = central, 4 = west

        # response vars
        totalComputers = get_score(line[16-19])
        computersForInstruction = get_score(line[24-27])
        integration = get_score(line[135]) # on a range of 1-4 if district staff help technology integration
        training = get_score(line[154])

        # add to processed_data
        dist_data = [idn, dist_size, urb, region, totalComputers, computersForInstruction, integration, training]
        processed_data.append(dist_data)



# no nulls in data

print(len(processed_data)) 
print(len(processed_data[0]))
