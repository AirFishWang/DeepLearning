# -*- coding: utf-8 -*-
def replace_tab(input_file, output_file, tab_size):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    lines = [line.replace('\t', ' ' * tab_size) for line in lines]
    with open(output_file, 'w') as f:
        f.writelines(lines)


if __name__ == "__main__":
    input_file = "/home/wangchun/Desktop/tab_test"
    output_file = "/home/wangchun/Desktop/tab_test_out"
    replace_tab(input_file, output_file, 4)