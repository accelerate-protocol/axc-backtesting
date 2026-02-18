#!/usr/bin/env python3
import sys
import os
import argparse
import logging

class MarkdownFileSplitter:
    """
    A class to process a file stream, splitting it based on delimiters.
    'process()' handles the cleanup internally.
    'exit()' is a NOP to allow nesting context managers.
    """

    def __init__(self, input_stream, delimiter, overwrite=False):
        """
        Initialize the splitter.
        
        :param input_stream: The file-like object to read from.
        :param delimiter: The string identifying a new file boundary.
        """
        self._input = input_stream
        self._delimiter = delimiter
        self._overwrite = overwrite

    def process(self):
        """
        Reads from the input stream and writes to subfiles.
        Current file handling and closing are local to this method.
        """
        current_file = None
        should_unescape = False

        try:
            for line in self._input:
                if line.startswith(self._delimiter):
                    # Close previous file if it exists
                    if current_file is not None:
                        current_file.close()
                    
                    current_file = None

                    # Extract filename and remove backslashes
                    filename = line[len(self._delimiter):].strip()
                    filename = filename.replace('\\', '')

                    if not self._overwrite and os.path.exists(filename):
                        logging.warning(f"Output file '{filename}' already exists. Skipping.")
                        continue
                elif current_file is None and line.strip():
                    # Open new file
                    current_file = open(filename, 'w', encoding='utf-8')
                    # Set local setting
                    should_unescape = not filename.endswith('.md')

                if current_file is not None:
                    # Write content
                    content = line
                    if should_unescape:
                        # Remove backslashes
                        content = content.replace('\\', '')
                    current_file.write(content)
                    
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            raise
        finally:
            # Ensure file is closed at the end of processing
            if current_file is not None:
                current_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="Split files using a delimiter.")
    
    # Positional argument for input file. 
    parser.add_argument('input_file', nargs='?', help='Input file path (default: stdin)')
    
    # Optional argument for delimiter
    parser.add_argument('--delimiter', '-d', default='---', help='Delimiter line to split files')
    parser.add_argument("--overwrite", action="store_const", const=True, default=False)
    
    args = parser.parse_args()

    # Handle Input Stream
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                splitter = MarkdownFileSplitter(f, args.delimiter, args.overwrite)
                splitter.process()
        except FileNotFoundError:
            logging.error(f"File not found: '{args.input_file}'")
            sys.exit(1)
    else:
        # Default to stdin
        splitter = MarkdownFileSplitter(sys.stdin, args.delimiter)
        splitter.process()

if __name__ == "__main__":
    main()
