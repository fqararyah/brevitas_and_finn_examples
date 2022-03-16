import json

def sum_resources(file_name):
    f = open(file_name)
    overall_resources = {}
    data = json.load(f)
    print(data['ConvolutionInputGenerator_0']['LUT'])
    for _,dict in data.items():
        for resource, consumption in dict.items():
            if resource not in overall_resources:
                overall_resources[resource] = 0
            overall_resources[resource] += int(consumption)
    print(overall_resources)
    with open(file_path + 'overall_resources.json', 'w') as fp:
        json.dump(overall_resources, fp)
    f.close()

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_96_fps_5_ns/reports/report/'
file_name = 'estimate_layer_resources_hls.json'

sum_resources(file_path + file_name)

file_path = '/home/fareed/wd/finn2/finn/my_prjects/brevitas_and_finn/getting_started/out/finn_out/mobilenet_zcu_102_192_fps_5_ns/reports/report/'

sum_resources(file_path + file_name)
