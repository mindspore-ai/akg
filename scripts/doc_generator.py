import os
import argparse
import yaml
import re
import unicodedata

def get_display_width(s):
    """
    Calculates the display width of a string, treating CJK characters as width 2.
    Uses unicodedata for more accurate character width calculation.
    """
    width = 0
    for char in s:
        # Use unicodedata to get character category
        category = unicodedata.category(char)
        # Check if it's a wide character (CJK, fullwidth, etc.)
        if unicodedata.east_asian_width(char) in ('F', 'W'):
            width += 2
        elif category.startswith('M'):  # Mark characters (combining)
            width += 0
        else:
            width += 1
    return width

class LiteralString(str):
    pass

def literal_presenter(dumper, data):
    """
    Custom YAML presenter for literal block scalars.
    """
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')

# Add the custom presenter to PyYAML for proper literal block formatting.
yaml.add_representer(LiteralString, literal_presenter, Dumper=yaml.SafeDumper)

class DocGenerator:
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def _format_table(self, table_markdown):
        """
        Parses a markdown table and formats it with proper alignment for mixed CJK/ASCII text.
        """
        lines = table_markdown.strip().split('\n')
        if len(lines) < 2:
            return table_markdown

        # Parse header
        header_line = lines[0].strip()
        if not header_line.startswith('|') or not header_line.endswith('|'):
            return table_markdown
            
        header = [h.strip() for h in header_line.strip('|').split('|')]
        
        # Validate separator line
        separator = lines[1].strip()
        if not re.match(r'^[|:\-\s]+$', separator):
            return table_markdown

        # Parse data rows
        rows = []
        for line in lines[2:]:
            line = line.strip()
            if line.startswith('|') and line.endswith('|'):
                row = [r.strip() for r in line.strip('|').split('|')]
                rows.append(row)

        # Ensure all rows have the same number of columns as the header
        num_columns = len(header)
        for i, row in enumerate(rows):
            if len(row) > num_columns:
                rows[i] = row[:num_columns]
            elif len(row) < num_columns:
                rows[i].extend([''] * (num_columns - len(row)))
        
        # Calculate max width for each column
        col_widths = [0] * num_columns
        all_rows = [header] + rows
        
        for row in all_rows:
            for i, cell in enumerate(row):
                if i < num_columns:
                    width = get_display_width(cell)
                    col_widths[i] = max(col_widths[i], width)

        # Build the formatted table
        formatted_lines = []
        
        # Format header
        header_parts = []
        for i, cell in enumerate(header):
            cell_width = get_display_width(cell)
            padding = col_widths[i] - cell_width
            header_parts.append(f" {cell}{' ' * padding} ")
        formatted_lines.append('|' + '|'.join(header_parts) + '|')

        # Format separator
        separator_parts = []
        for width in col_widths:
            separator_parts.append('-' * (width + 2))  # +2 for spaces around content
        formatted_lines.append('|' + '|'.join(separator_parts) + '|')

        # Format data rows
        for row in rows:
            row_parts = []
            for i, cell in enumerate(row):
                if i < num_columns:
                    cell_width = get_display_width(cell)
                    padding = col_widths[i] - cell_width
                    row_parts.append(f" {cell}{' ' * padding} ")
            formatted_lines.append('|' + '|'.join(row_parts) + '|')

        return '\n'.join(formatted_lines)

    def process_file(self, filename):
        """
        Processes a single markdown file, formats tables, and generates a YAML file.
        """
        op_name = os.path.splitext(filename)[0]
        src_path = os.path.join(self.src_dir, filename)
        dest_path = os.path.join(self.dest_dir, f"{op_name}_doc.yaml")

        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Use a state machine to find and format all markdown tables
            processed_content = []
            table_buffer = []
            in_table = False
            lines = content.split('\n')

            for line in lines:
                is_table_line = line.strip().startswith('|') and line.strip().endswith('|')
                
                if is_table_line:
                    if not in_table:
                        in_table = True
                    table_buffer.append(line)
                else:
                    if in_table:
                        # A table block has just ended, process it.
                        is_valid_table = len(table_buffer) > 1 and re.match(r'^[|:\-\s]+$', table_buffer[1].strip())
                        if is_valid_table:
                            formatted_table = self._format_table('\n'.join(table_buffer))
                            processed_content.append(formatted_table)
                        else:
                            processed_content.extend(table_buffer)
                        
                        table_buffer = []
                        in_table = False
                    
                    processed_content.append(line)
            
            # If the file ends with a table, process the last buffer.
            if in_table and table_buffer:
                is_valid_table = len(table_buffer) > 1 and re.match(r'^[|:\-\s]+$', table_buffer[1].strip())
                if is_valid_table:
                    formatted_table = self._format_table('\n'.join(table_buffer))
                    processed_content.append(formatted_table)
                else:
                    processed_content.extend(table_buffer)

            final_content = '\n'.join(processed_content)

            # Use LiteralString to ensure the output YAML uses the literal block style `|`
            yaml_data = {
                op_name: {
                    'description': LiteralString(final_content)
                }
            }

            with open(dest_path, 'w', encoding='utf-8') as f:
                yaml.dump(yaml_data, f, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)

            print(f"Successfully generated and formatted {dest_path}")

        except Exception as e:
            print(f"Error processing {src_path}: {e}")

    def generate_all(self):
        """
        Generates YAML documentation for all markdown files in the source directory.
        Skips existing YAML files to avoid conflicts when src and dest are the same.
        """
        for filename in os.listdir(self.src_dir):
            if filename.endswith(".md") and not filename.endswith("_doc.yaml"):
                self.process_file(filename)

def main():
    parser = argparse.ArgumentParser(description="Generate YAML documentation from Markdown files with table formatting.")
    parser.add_argument("--src_dir", type=str, default="yaml/doc", help="Source directory containing Markdown files.")
    parser.add_argument("--dest_dir", type=str, default="yaml/doc", help="Destination directory for generated YAML files.")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir_abs = os.path.join(base_dir, args.src_dir)
    dest_dir_abs = os.path.join(base_dir, args.dest_dir)

    generator = DocGenerator(src_dir_abs, dest_dir_abs)
    generator.generate_all()

if __name__ == "__main__":
    main()