#!/usr/bin/env python3
import sys
import os
import argparse
import logging
import re
from typing import Optional, IO

class MarkdownFileSplitter:
    """
    A class to process a file stream, splitting it based on delimiters.
    'process()' handles the cleanup internally.
    'exit()' is a NOP to allow nesting context managers.
    """

    def __init__(self, input_stream: IO, delimiter: str, overwrite: bool=False):
        """
        Initialize the splitter.
        
        :param input_stream: The file-like object to read from.
        :param delimiter: The string identifying a new file boundary.
        """
        self._input = input_stream
        self._delimiter = delimiter
        self._overwrite = overwrite

    def process(self) -> None:
        """
        Reads from the input stream and writes to subfiles.
        Current file handling and closing are local to this method.
        """
        current_file: Optional[IO] = None
        should_unescape: bool = False
        lines: list[str] = list(self._input)
        images_dict: dict[str, str] = {}
        pending_images: list[str] = []
        content : list[str]= []

        try:
            for line in lines:
                if match := re.match(r"\[([^]]+)\]\s*:\s*(.*)", line):
                    images_dict[match.group(1)] = match.group(2)
                else:
                    content.append(line)

            for line in content:
                if line.startswith(self._delimiter):
                    # Close previous file if it exists
                    if current_file is not None:
                        for image in pending_images:
                            if image in images_dict:
                                current_file.write(f'[{image}]: {images_dict[image]}\n')
                        current_file.close()
                    current_file = None
                    pending_images = []

                    # Extract filename and remove backslashes
                    filename = line[len(self._delimiter):].strip().replace('\\', '')

                    if not self._overwrite and os.path.exists(filename):
                        logging.warning(f"Output file '{filename}' already exists. Skipping.")
                        continue
                elif current_file and line.strip():
                    # Open new file
                    current_file = open(filename, 'w', encoding='utf-8')
                    # Set local setting
                    should_unescape = not filename.endswith('.md')
                elif match := re.search(r"!\[\]\[([^]]+)\]\s*", line):
                    pending_images.append(match.group(1))

                if current_file:
                    current_file.write(
                        line.replace('\\', '') \
                        if should_unescape else line
                    )
                    
        except Exception as e:
            logging.error(f"An error occurred during processing: {e}")
            raise
        finally:
            # Ensure file is closed at the end of processing
            if current_file:
                for image in pending_images:
                    if image in images_dict:
                        current_file.write(f'[{image}]: {images_dict[image]}\n')
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
