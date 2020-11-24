import os
import pathlib

build_path = '_build/docx'

pathlib.Path(build_path).mkdir(parents=True, exist_ok=True)

idx_file = open('contents.inc', 'r')
within_toc_block = False
build_files = []
command = 'pandoc -o {0} -f rst+east_asian_line_breaks -s {1}'

for line in idx_file:
    if within_toc_block == False:
        if line.startswith('.. toctree::'):
            within_toc_block = True
    else:
        if line.startswith('   :'):
            continue
        elif not line.strip(' '):
            continue
        elif line.startswith('  ') and line.strip():
            build_files.append(line.strip())

file_args = []

for i, f in enumerate(build_files):
    file_args.append(f + '.rst')
    output_file = os.path.join(build_path, '{0}-{1}.docx'.format(i, f))
    os.system(command.format(output_file, f + '.rst'))
    print('{0} converted successfully'.format(f))

os.system(command.format(
    os.path.join(build_path, 'all-in-one.docx'), ' '.join(file_args)))
print('all-in-one converted successfully')