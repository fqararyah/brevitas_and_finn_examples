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

file_path = '/home/fareed/wd/my_repos/brevitas_and_finn/getting_started/out/finn_out/mobilenet/reports/report/'
file_name = 'estimate_layer_resources_hls.json'

sum_resources(file_path + file_name)