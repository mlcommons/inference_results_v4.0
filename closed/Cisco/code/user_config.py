#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import configparser
import os, sys
import argparse

def main(argv, argc):
    config_custom = configparser.ConfigParser()
    config_custom.optionxform = str
    config_custom['custom'] = {}
    output_file = "user.conf"
    ci_run = 0
    config = configparser.ConfigParser()
    config.optionxform = str

    if argc > 1:
        config_file = argv[1]
        if argc > 2:
            output_file = argv[2]
    else:
        config_file = 'user_default.conf'

    print(config_file)
    config.read(config_file)

    if os.getenv('number_cores') is not None:
        number_cores = int(os.getenv('number_cores'))
    elif os.getenv('num_physical_cores') is not None:
        number_cores = int(os.getenv('num_physical_cores'))
    else:
        number_cores = os.cpu_count()
    if os.getenv('ci_run') is not None:
        ci_run = int(os.getenv('ci_run'))

    default_number_cores = int(config.get('default','number_cores'))

    for section in config.sections():
        if section == 'ci' and ci_run != 1:
            continue
        print('\n section  %s' % (section))
        if config.has_section(section):
            for name, value in config.items(section):
                if name == "number_cores":
                    continue

                print('default  %s = %s' % (name, value))

                if name.find("_qps") == -1:
                    custom_value = value
                else:
                    custom_value = str(round(float(value) * (number_cores / default_number_cores), 4))

                config_custom['custom'][name] = custom_value
                print('custom  %s = %s' % (name, custom_value))


    with open(output_file, 'w') as f:
        config_custom.write(f)

    with open(output_file, 'r') as fin:
        data = fin.read().splitlines(True)

    with open(output_file, 'w') as fout:
        fout.writelines(data[1:])


if __name__ == '__main__':
    main(sys.argv, len(sys.argv))

