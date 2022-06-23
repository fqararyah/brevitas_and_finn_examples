import json
import matplotlib.ticker as mtick

def sum_resources(file_name):
    f = open(file_name)
    overall_resources = {}
    data = json.load(f)
    print(data['ConvolutionInputGenerator_0']['LUT'])
    for _, dict in data.items():
        for resource, consumption in dict.items():
            if resource not in overall_resources:
                overall_resources[resource] = 0
            overall_resources[resource] += int(consumption)
    print(overall_resources)
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
        for i in range(int(iddleness_percentage * 100 + 1)):
            iddleness_dict[i] += 100 * int(layers_resources[layer_name]['LUT']) / overall_resources['LUT']

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

print('weighted imbalance: ', iddleness_dict)
 
ax1.plot( list(iddleness_dict.keys()), list(iddleness_dict.values()), label= 'MobileNetV1')
ax1.scatter(list(iddleness_dict.keys()), list(iddleness_dict.values()) )

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

print('weighted imbalance: ', iddleness_dict)
 
ax1.plot( list(iddleness_dict.keys()), list(iddleness_dict.values()), label= 'MobileNetV1_0.5' )
ax1.scatter(list(iddleness_dict.keys()), list(iddleness_dict.values()) )

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