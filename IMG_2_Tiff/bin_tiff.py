import os
import subprocess
file_list = []
for f in os.listdir('.'):
    if os.path.splitext(f)[1] == '.img':
        file_list.append(f)
    print(file_list)
for f in file_list:    
    out_filename = './out/' + os.path.splitext(f)[0] + '.tif'
    command_str = 'gdal_translate {in_filename} {out_filename}'.format(in_filename=f, out_filename=out_filename)
    print(command_str)
    p = subprocess.Popen(command_str)
    p.wait()
