import re
import numpy as np
import matplotlib.pyplot as plt

def xvg_reader(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    data_lines = []
    for line in lines:
        if re.match(r'^\s*\d', line):
            data_lines.append(line)

    matrix = []
    for data_line in data_lines:
        row = []
        for value in data_line.split():
            row.append(float(value))
        matrix.append(row)

    return np.array(matrix)



rdf1 = xvg_reader('E:\Postdoc\ionic_liquid\simulations\shear\AA\\2x3x3_256iop_20ns\set-ups\\1e-2\PF6-PF6.xvg')
#rdf2 = xvg_reader('E:\Postdoc\ionic_liquid\PF6-PF6_80_notnormalized.xvg')

plt.plot(rdf1[:,0], rdf1[:,1])
#plt.plot(rdf2[:,0], rdf2[:,1])

plt.show()