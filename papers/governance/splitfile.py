#!/usr/bin/env python3
"""
splitfile.py - splits a markdown file into subfiles
"""

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
        self._input: IO = input_stream
        self._delimiter: str = delimiter
        self._overwrite: bool = overwrite
        self._current_file: Optional[IO] = None
        self._images_dict: dict[str, str] = {}
        self._pending_images: list[str] = []


    def process(self) -> None:
        """
        Reads from the input stream and writes to subfiles.
        Current file handling and closing are local to this method.
        """
        should_unescape: bool = False
        lines: list[str] = list(self._input)
        content : list[str]= []
        filename: Optional[str] = None

        self._current_file = None
        self._images_dict = {}
        self._pending_images = []
        try:
            for line in lines:
                if match := re.match(r"\[([^]]+)\]\s*:\s*(.*)", line):
                    self._images_dict[match.group(1)] = match.group(2)
                else:
                    content.append(line)

            for line in content:
                if line.startswith(self._delimiter):
                    # Close previous file if it exists
                    self.__close_file()

                    # Extract filename and remove backslashes
                    filename = line[len(self._delimiter):].strip().replace('\\', '')

                    if not self._overwrite and os.path.exists(filename):
                        logging.warning(
                            "Output file '%s' already exists. Skipping.", filename
                        )
                        filename = None
                    continue

                # skip white space headers
                if filename and not self._current_file and line.strip():
                    # Open new file
                    self._current_file = open(filename, 'w', encoding='utf-8')
                    # Set local setting
                    should_unescape = not filename.endswith('.md')

                if self._current_file:
                    if match := re.search(r"!\[\]\[([^]]+)\]\s*", line):
                        self._pending_images.append(match.group(1))

                    self._current_file.write(
                        line.replace('\\', '') \
                        if should_unescape else line
                    )
        except Exception:
            logging.error("An error occurred during processing", exc_info=True)
            raise
        finally:
            self.__close_file()

    def __close_file(self) -> None:
        """ Dump out image data and close file """
        if self._current_file:
            for image in self._pending_images:
                if image in self._images_dict:
                    self._current_file.write(f'[{image}]: {self._images_dict[image]}\n')
            self._current_file.close()
        self._current_file = None
        self._pending_images = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return True

def main():
    """
    main function
    """
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
            with open(args.input_file, 'r', encoding='utf-8') as f:
                splitter = MarkdownFileSplitter(f, args.delimiter, args.overwrite)
                splitter.process()
        except FileNotFoundError:
            logging.error("File not found: '%s'", args.input_file)
            sys.exit(1)
    else:
        # Default to stdin
        splitter = MarkdownFileSplitter(sys.stdin, args.delimiter)
        splitter.process()

if __name__ == "__main__":
    main()
