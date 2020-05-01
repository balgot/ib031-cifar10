import nbformat as nbf
import sys, os

CODE = 'code'
MARKDOWN = 'markdown'

nb = nbf.v4.new_notebook()

def get_type(line: str) -> str:
    line = line.lstrip()
    if line.startswith('##'):
        return MARKDOWN
    return CODE

def add_cell(cell_buffer: str, cell_type: str) -> str:
    cell_buffer = cell_buffer.strip()
    if cell_buffer:
        if cell_type == CODE:
            nb['cells'].append(nbf.v4.new_code_cell(cell_buffer))
        elif cell_type == MARKDOWN:
            nb['cells'].append(nbf.v4.new_markdown_cell(cell_buffer))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], 'input_file', '[output_file]')
        print('Output_file should end with ".ipynb". If it doesn\'t, this '
              'extension is added.')
        exit(1)

    with open(sys.argv[1], 'r') as fin:
        cell_type = None
        cell_buffer = ''

        for line in fin:
            line_type = get_type(line)
            if line_type == MARKDOWN:
                line = line.lstrip()[2:]

            if cell_type == line_type:
                cell_buffer += line
            else:
                add_cell(cell_buffer, cell_type)
                cell_buffer = line
                cell_type = line_type
        add_cell(cell_buffer, cell_type)

    if len(sys.argv) == 2:
        fout = os.path.splitext(sys.argv[1])[0] + '.ipynb'
    else:
        fout = sys.argv[2]
        if not fout.endswith('.ipynb'):
            fout += '.ipynb'

    nbf.write(nb, fout)
