import csv


num_queries = 100

'''
'emissions.csv'
'emissions_100.csv'
'emissions_10.csv'
'''
# read output  
with open('emissions_100.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

print("Data:\n", data)


# get duration and energy 
duration = float(data[1][3])
energy_kWh = float(data[1][5])

print("Duration: {} s \nEnergy: {} kWh".format(duration, energy_kWh))


# convert energy from kWh to Joules 
ratio = 3.6e6
energy_J = energy_kWh * ratio
print("Energy: {} joules over all query images".format(energy_J))

energy_J_per_query = energy_J / num_queries
print("Energy: {} joules over one query image".format(energy_J_per_query))


# find power in Watts

duration_h = duration / 3600 
power_W = 1000 * (energy_kWh / duration_h)

print("Power in Watts: {}".format(power_W))

duration_per_query = duration / 100
energy_from_power = power_W * duration_per_query

print("Duration: {} s per query\nenergy: {} J per query\n".format(duration_per_query, energy_J_per_query))


power_consumption = energy_J_per_query / duration_per_query
print("Power consumption per query image: {}".format(power_consumption))
