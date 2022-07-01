import json
import matplotlib.ticker as mtick

def sum_resources(file_name):
    f = open(file_name)
    overall_resources = {}
    data = json.load(f)
    for _, dict in data.items():
        for resource, consumption in dict.items():
            if resource not in overall_resources:
                overall_resources[resource] = 0
            overall_resources[resource] += int(consumption)
    with open(file_path + 'overall_resources.json', 'w') as fp:
        json.dump(overall_resources, fp)
    f.close()
    return overall_resources


def get_layers_cycles(file_name):
    f = open(file_name)
    layers_cycles = {}
    layers_cycles = json.load(f)
    return layers_cycles

def get_layers_resources(file_name):
    f = open(file_name)
    layers_resources = {}
    layers_resources = json.load(f)
    return layers_resources

def get_overall_cycles(file_name):
    f = open(file_name)
    rtl_sim = {}
    rtl_sim = json.load(f)
    return int(rtl_sim['cycles'])

import matplotlib.pyplot as plt

def get_iddleness_dict(layers_cycles, layers_resources, overall_resources):
    max_cycles = 0
    for layer_name, cycles in layers_cycles.items():
        if cycles > max_cycles:
            max_cycles = cycles

    iddleness_dict = {}
    for i in range(100):
        iddleness_dict[i] = 0

    for layer_name, cycles in layers_cycles.items():
        iddleness_percentage = (max_cycles - cycles) / max_cycles
        for i in range(int(iddleness_percentage * 100)):
            iddleness_dict[i] += 100 * int(layers_resources[layer_name]['LUT']) / overall_resources['LUT']

    return iddleness_dict

def fibha_iddleness_v1():
    all_pipe_iddle = 27 #%
    iddleness_dict = {}
    pipe_iddleness_dict = {0: 0.17, 1:0.071, 2: 0.14, 3:0.46,4:0.54,5:0,6:0.14}
    resource_dict = {0: 54, 1:18, 2: 128, 3:9,4:128,5:18,6:256}
    sum_pipe_resources = 611
    over_all_resources = 4096
    
    for i in range(all_pipe_iddle):
        iddleness_dict[i] = 100 * sum_pipe_resources / over_all_resources
    
    for i in range(all_pipe_iddle, 100):
        for key, val in pipe_iddleness_dict.items():
            if val >= (i - all_pipe_iddle) / 100:
                if i not in iddleness_dict:
                    iddleness_dict[i] = 0
                iddleness_dict[i] += 100 * resource_dict[key] / over_all_resources
    
    return iddleness_dict

def fibha_iddleness_v1_0_5():
    all_pipe_iddle = 41 #%
    iddleness_dict = {}
    pipe_iddleness_dict = {0: 0.17, 1:0.071, 2: 0.14, 3:0.46,4:0.54,5:0,6:0.14}
    resource_dict = {0: 54*2, 1:18, 2: 128, 3:9,4:128,5:18,6:256}
    sum_pipe_resources = 611 + 54
    over_all_resources = 4096
    
    for i in range(all_pipe_iddle):
        iddleness_dict[i] = 100 * sum_pipe_resources / over_all_resources
    
    for i in range(all_pipe_iddle, 100):
        for key, val in pipe_iddleness_dict.items():
            if val >= (i - all_pipe_iddle) / 100:
                if i not in iddleness_dict:
                    iddleness_dict[i] = 0
                iddleness_dict[i] += 100 * resource_dict[key] / over_all_resources
    
    return iddleness_dict

fig, ax1 = plt.subplots()
finn_resource_utilization = []
comp_resources = 274080

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_220_fps_5_ns/reports/report/'
file_name = 'estimate_layer_resources_hls.json'

sum_resources(file_path + file_name)

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_6_fps_5_ns_0_5/reports/report/'

sum_resources(file_path + file_name)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_220_fps_5_ns/reports/report/estimate_layer_cycles.json'

layers_cycles = get_layers_cycles(file_name)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_220_fps_5_ns/reports/report/estimate_layer_resources_hls.json'

layers_resources = get_layers_resources(file_name)
overall_resources = sum_resources(file_name)
finn_resource_utilization.append(overall_resources['LUT'] / comp_resources)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_220_fps_5_ns/reports/report/rtlsim_performance.json'

overall_cycles = get_overall_cycles(file_name)

iddleness_dict = get_iddleness_dict(layers_cycles, layers_resources, overall_resources)

fibha_iddleness = fibha_iddleness_v1()

 
ax1.plot( list(iddleness_dict.keys()), list(iddleness_dict.values()), label= 'FINN_V1')
#ax1.scatter(list(iddleness_dict.keys()), list(iddleness_dict.values()) )

ax1.plot( list(fibha_iddleness.keys()), list(fibha_iddleness.values()), label= 'FiBHA_V1')
#ax1.scatter(list(fibha_iddleness.keys()), list(fibha_iddleness.values()) )
##############################################################################3

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_440_fps_5_ns_0_5/reports/report/'
file_name = 'estimate_layer_resources_hls.json'

sum_resources(file_path + file_name)

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_440_fps_5_ns_0_5/reports/report/'

sum_resources(file_path + file_name)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_440_fps_5_ns_0_5/reports/report/estimate_layer_cycles.json'

layers_cycles = get_layers_cycles(file_name)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_440_fps_5_ns_0_5/reports/report/estimate_layer_resources_hls.json'

layers_resources = get_layers_resources(file_name)
overall_resources = sum_resources(file_name)
finn_resource_utilization.append(overall_resources['LUT'] / comp_resources)

file_name = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_440_fps_5_ns_0_5/reports/report/rtlsim_performance.json'

overall_cycles = get_overall_cycles(file_name)

iddleness_dict = get_iddleness_dict(layers_cycles, layers_resources, overall_resources)
fibha_iddleness_0_5 = fibha_iddleness_v1_0_5()
 
ax1.plot( list(iddleness_dict.keys()), list(iddleness_dict.values()), label= 'FINN_V1_0.5' )
ax1.plot( list(fibha_iddleness_0_5.keys()), list(fibha_iddleness_0_5.values()), label= 'FiBHA_V1_0.5')

ax1.grid(axis='y', linestyle='-')
ax1.set_axisbelow(True)
#ax1.scatter(list(iddleness_dict.keys()), list(iddleness_dict.values()) )

plt.xlabel('Time')
ax1.set_xlabel('Time')
ax1.set_ylabel('Idle Compute Resources')

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as mtick


fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
ax1.xaxis.set_major_formatter(xticks)
ax1.yaxis.set_major_formatter(xticks)

plt.legend()

plt.savefig( './' + 'weighted_iddleness.pdf')
plt.savefig( './' + 'weighted_iddleness.png')

fibha_resource_utilization = [(178031 + 41149) / 274080, (90627 + 142354) / 274080]

fig, ax = plt.subplots()
on_x = ['MobileNetV1', 'MobileNetV1_0.5']
width = [0.35] * len(on_x)
x_coordinates_0 = [i for i in range(len(on_x))]
x_coordinates_1 = [i - width[i] / 2 for i in range(len(on_x))]
x_coordinates_2 = [i + width[i] / 2 for i in range(len(on_x))]

print(finn_resource_utilization)
on_y_1 = finn_resource_utilization
on_y_2 = fibha_resource_utilization

bars_1 = ax.bar(x_coordinates_1, on_y_1, width, label='FINN')
bars_2 = ax.bar(x_coordinates_2, on_y_2, width, label='FiBHA')

ax.grid(axis='y', linestyle='-')
ax.set_xticks(x_coordinates_0, on_x)

ax.legend()

plt.gcf().subplots_adjust(bottom=0.15)
plt.gcf().subplots_adjust(left=0.15)
plt.ylabel('Resource Utilization', fontsize=16)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(16)

plt.savefig( './' + 'resource_utilization.pdf')
plt.savefig( './' + 'resource_utilization.png')
